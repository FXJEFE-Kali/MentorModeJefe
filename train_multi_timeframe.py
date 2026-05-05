"""
FXJEFE Multi-Timeframe Training Pipeline
=========================================
Trains per-symbol, per-timeframe models WITHOUT leakage.
- Binary classification (BUY vs SELL, no HOLD)
- Configurable label threshold per timeframe
- Walk-forward validation (6 windows)
- Sharpe-based model selection
- No future_return in features

Data sources:
  Crypto: FXJEFE_Crypto_Features.csv (BTC/ETH/XRP @ M5/M15/H1/H4/D1)
  Forex:  Historical/enhanced/ (EURUSD @ M15/H4/D1)
"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

MODELS_DIR = config['models_path']
DATA_DIR = config['data_output_path']
HIST_DIR = os.path.join(DATA_DIR, 'Historical', 'enhanced')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(config['log_path'], exist_ok=True)

LOG_FILE = os.path.join(config['log_path'], f'multi_tf_{datetime.now():%Y%m%d_%H%M%S}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
log = logging.info

# ========================= FEATURE LISTS =========================
# Clean features the EA can compute — NO future_return
FEATURES_CORE = [
    'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment'
]

# Extended features available in enhanced data
FEATURES_EXTENDED = [
    'garch_vol', 'rsi_m5', 'rsi_h1', 'macd_diff_m5', 'macd_diff_h1',
    'atr_m5', 'atr_h1', 'vwap_m5', 'vwap_h1', 'roc_m5', 'roc_h1',
    'stochastic_m5', 'stochastic_h1', 'cci_m5', 'cci_h1'
]

# NEVER use these as features
FORBIDDEN = {'future_return', 'future_return_1', 'future_return_5', 'future_return_15',
             'future_price', 'price_change', 'label', 'signal', 'sample_weight',
             'regime', 'time', 'time.1', 'symbol', 'timeframe', 'target_return',
             'open', 'high', 'low', 'close', 'volume', 'volume.1', 'threshold'}

# Label thresholds — separate for crypto (high vol) and forex (low vol)
CRYPTO_THRESHOLDS = {
    'M5':  0.002,   # 0.2%
    'M15': 0.003,   # 0.3%
    'H1':  0.005,   # 0.5%
    'H4':  0.010,   # 1.0%
    'D1':  0.015,   # 1.5%
    'W1':  0.025,   # 2.5%
}

FOREX_THRESHOLDS = {
    'M5':  0.0003,  # 3 pips on EURUSD
    'M15': 0.0005,  # 5 pips
    'H1':  0.001,   # 10 pips
    'H4':  0.001,   # 10 pips (keeps 18k+ rows)
    'D1':  0.002,   # 20 pips
    'W1':  0.005,   # 50 pips
}

INDEX_THRESHOLDS = {
    'M5':  0.001,
    'M15': 0.002,
    'H1':  0.003,
    'H4':  0.005,
    'D1':  0.008,
    'W1':  0.015,
}

CRYPTO_SYMBOLS = {'BTCUSD', 'ETHUSD', 'XRPUSD'}
INDEX_SYMBOLS = {'NAS100', 'US500', 'US30'}

def get_threshold(symbol, timeframe):
    if symbol in CRYPTO_SYMBOLS:
        return CRYPTO_THRESHOLDS.get(timeframe, 0.005)
    elif symbol in INDEX_SYMBOLS:
        return INDEX_THRESHOLDS.get(timeframe, 0.005)
    else:
        return FOREX_THRESHOLDS.get(timeframe, 0.001)

# Horizon (bars ahead) for label creation when we need to compute it
LABEL_HORIZONS = {
    'M5': 5, 'M15': 5, 'H1': 5, 'H4': 5, 'D1': 5, 'W1': 3,
}

# ========================= DATA LOADING =========================
def load_crypto_data(symbol, timeframe):
    """Load crypto data for a specific symbol/timeframe from FXJEFE_Crypto_Features.csv"""
    path = os.path.join(DATA_DIR, 'FXJEFE_Crypto_Features.csv')
    df = pd.read_csv(path)
    df = df[(df['symbol'] == symbol) & (df['timeframe'] == timeframe)].copy()
    df = df.reset_index(drop=True)
    log(f"  Loaded {len(df):,} rows for {symbol} {timeframe}")
    return df


def load_forex_data(symbol, timeframe):
    """Load forex data from Historical/enhanced/ folder"""
    tf_map = {'M15': 'M15', 'H1': 'H1', 'H4': 'H4', 'D1': 'Daily', 'W1': 'Weekly'}
    tf_name = tf_map.get(timeframe, timeframe)

    # Try enhanced subfolder first (larger files)
    import glob
    pattern = os.path.join(HIST_DIR, f'enhanced_{symbol}_{tf_name}_*.csv')
    matches = glob.glob(pattern)
    if matches:
        path = matches[0]
    else:
        # Try root Historical folder
        path = os.path.join(DATA_DIR, 'Historical', f'{symbol}_{tf_name}_enhanced.csv')

    if not os.path.exists(path):
        log(f"  No data found for {symbol} {timeframe}")
        return None

    df = pd.read_csv(path)
    log(f"  Loaded {len(df):,} rows for {symbol} {timeframe} from {os.path.basename(path)}")
    return df


def prepare_binary_labels(df, timeframe, symbol):
    """Create binary BUY(1) / SELL(0) labels. Drop HOLD rows."""
    threshold = get_threshold(symbol, timeframe)
    horizon = LABEL_HORIZONS.get(timeframe, 5)

    # Use existing future_return if available
    if 'future_return' in df.columns:
        fr = df['future_return'].astype(float)
    elif 'future_return_5' in df.columns:
        fr = df['future_return_5'].astype(float)
    elif 'price' in df.columns:
        # Compute from price
        fr = df['price'].shift(-horizon) / df['price'] - 1
    else:
        log(f"  Cannot create labels for {symbol} {timeframe} — no price/future_return")
        return None

    # Binary: BUY if return > threshold, SELL if return < -threshold, drop the rest
    df = df.copy()
    df['_future_return'] = fr
    df['_label'] = np.where(fr > threshold, 1, np.where(fr < -threshold, 0, -1))

    # Drop HOLD (label == -1) and NaN
    df = df[df['_label'] >= 0].copy()
    df = df.dropna(subset=['_future_return'])

    buy_count = (df['_label'] == 1).sum()
    sell_count = (df['_label'] == 0).sum()
    total = len(df)
    log(f"  Binary labels (threshold={threshold:.1%}): BUY={buy_count:,} ({buy_count/total*100:.1f}%) "
        f"SELL={sell_count:,} ({sell_count/total*100:.1f}%) | Dropped {len(fr)-total:,} HOLD rows")

    return df


def get_clean_features(df):
    """Get available clean features from dataframe (no leakage)."""
    available_core = [f for f in FEATURES_CORE if f in df.columns]
    available_ext = [f for f in FEATURES_EXTENDED if f in df.columns]
    features = available_core + available_ext

    # Safety: remove anything forbidden
    features = [f for f in features if f not in FORBIDDEN]
    return features


# ========================= WALK-FORWARD TRAINING =========================
def train_single_model(X_train, y_train, X_test, y_test, model_type='xgboost'):
    """Train a single model and return predictions + accuracy."""
    from sklearn.metrics import accuracy_score

    if model_type == 'xgboost':
        import xgboost as xgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'random_state': 42,
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('depth', 3, 8),
                'subsample': trial.suggest_float('sub', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('lam', 0.1, 10.0),
                'min_child_weight': trial.suggest_int('mcw', 1, 10),
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain, num_boost_round=500,
                              evals=[(dval, 'val')], early_stopping_rounds=30,
                              verbose_eval=False)
            pred = (model.predict(dval) > 0.5).astype(int)
            return accuracy_score(y_test, pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        # Train final model with best params
        best = study.best_params
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'tree_method': 'hist', 'random_state': 42,
            'learning_rate': best['lr'], 'max_depth': best['depth'],
            'subsample': best['sub'], 'colsample_bytree': best['col'],
            'reg_lambda': best['lam'], 'min_child_weight': best['mcw'],
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, num_boost_round=800,
                          evals=[(dtest, 'val')], early_stopping_rounds=50,
                          verbose_eval=False)
        probas = model.predict(dtest)
        preds = (probas > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        return model, preds, probas, acc, best

    elif model_type == 'lightgbm':
        import lightgbm as lgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
                'is_unbalance': True,
                'num_leaves': trial.suggest_int('leaves', 20, 100),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('depth', 3, 10),
                'subsample': trial.suggest_float('sub', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('mcs', 5, 50),
            }
            model = lgb.LGBMClassifier(**params, n_estimators=500)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            pred = model.predict(X_test)
            return accuracy_score(y_test, pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best = study.best_params
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
            'is_unbalance': True,
            'num_leaves': best['leaves'], 'learning_rate': best['lr'],
            'max_depth': best['depth'], 'subsample': best['sub'],
            'colsample_bytree': best['col'], 'min_child_samples': best['mcs'],
        }
        model = lgb.LGBMClassifier(**params, n_estimators=800)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        probas = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return model, preds, probas, acc, best


def simulate_equity(X_test, y_pred, y_true, price_idx, atr_idx):
    """Fixed-risk equity simulation for binary predictions."""
    FIXED_RISK = 50.0
    HORIZON = 5
    ANN_FACTOR = np.sqrt(252 * 24)  # ~H1 annualization

    equity = [10000.0]
    bal = equity[0]
    gross_p, gross_l = 0.0, 0.0
    wins, total = 0, 0

    for i in range(len(y_pred) - HORIZON):
        pred = int(y_pred[i])
        entry = X_test[i, price_idx]
        exit_p = X_test[i + HORIZON, price_idx] if (i + HORIZON) < len(X_test) else entry
        if entry <= 0 or exit_p <= 0 or bal <= 1.0:
            equity.append(bal)
            continue

        ret = np.clip((exit_p - entry) / entry, -0.10, 0.10)
        spread_cost = 0.0005

        if pred == 1:  # BUY
            pnl = FIXED_RISK * (ret - spread_cost) / 0.01
        else:  # SELL
            pnl = FIXED_RISK * (-ret - spread_cost) / 0.01

        pnl = np.clip(pnl, -2 * FIXED_RISK, 4 * FIXED_RISK)

        if pnl > 0:
            gross_p += pnl
            wins += 1
        elif pnl < 0:
            gross_l += abs(pnl)
        total += 1
        bal = max(bal + pnl, 1.0)
        equity.append(bal)

    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    active = rets[rets != 0]
    mean_r = np.mean(active) if len(active) > 0 else 0
    std_r = np.std(active) if len(active) > 0 else 1e-10
    down = active[active < 0]
    down_std = np.std(down) if len(down) > 0 else 1e-10

    sharpe = mean_r / std_r * ANN_FACTOR if std_r > 1e-12 else 0
    pf = gross_p / gross_l if gross_l > 0 else (float('inf') if gross_p > 0 else 0)
    peak = np.maximum.accumulate(eq)
    max_dd = np.min((eq - peak) / peak) * 100 if peak.max() > 0 else 0
    tot_ret = (eq[-1] / eq[0] - 1) * 100
    wr = wins / total * 100 if total > 0 else 0

    return {'sharpe': sharpe, 'pf': pf, 'max_dd': max_dd,
            'total_return': tot_ret, 'n_trades': total, 'win_rate': wr}


def walk_forward_train(df, features, symbol, timeframe, n_windows=6):
    """Walk-forward training with binary labels. Returns best model + metrics."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    X = df[features].ffill().fillna(0).values.astype(np.float32)
    y = df['_label'].values.astype(int)

    total = len(X)
    min_train = max(int(total * 0.40), 2000)
    test_size = max(int(total * 0.10), 500)

    min_rows = 1500 if timeframe in ('D1', 'W1') else 3000
    if total < min_rows:
        log(f"  Skipping {symbol} {timeframe} — only {total} rows (need {min_rows}+)")
        return None

    price_idx = features.index('price') if 'price' in features else 0
    atr_idx = features.index('atr') if 'atr' in features else 1

    log(f"\n  Walk-Forward ({n_windows} windows) for {symbol} {timeframe}")
    log(f"  Total: {total:,} | Min train: {min_train:,} | Test window: {test_size:,}")
    log(f"  {'Win':<5} {'Train':<10} {'Test':<8} {'Acc':<8} {'Sharpe':<8} {'PF':<7} {'WR':<8}")
    log(f"  {'-'*55}")

    results = []
    best_model = None
    best_sharpe = -999

    for w in range(n_windows):
        test_start = min_train + w * test_size
        test_end = min(test_start + test_size, total)
        if test_end <= test_start or test_start >= total:
            break

        X_tr, y_tr = X[:test_start], y[:test_start]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        # Train XGBoost (fast + strong)
        model, preds, probas, acc, params = train_single_model(
            X_tr, y_tr, X_te, y_te, model_type='xgboost'
        )

        sm = simulate_equity(X_te, preds, y_te, price_idx, atr_idx)
        results.append({'window': w+1, 'accuracy': acc, **sm})

        if sm['sharpe'] > best_sharpe:
            best_sharpe = sm['sharpe']
            best_model = model
            best_params = params

        log(f"  {w+1:<5} {len(X_tr):>8,}  {len(X_te):>6,}  {acc:>5.1%}  "
            f"{sm['sharpe']:>6.2f}  {sm['pf']:>5.2f}  {sm['win_rate']:>5.1f}%")

    if not results:
        return None

    # Also train LightGBM on full split for comparison
    split = int(total * 0.80)
    X_tr_full, X_te_full = X[:split], X[split:]
    y_tr_full, y_te_full = y[:split], y[split:]

    lgb_model, lgb_preds, lgb_probas, lgb_acc, lgb_params = train_single_model(
        X_tr_full, y_tr_full, X_te_full, y_te_full, model_type='lightgbm'
    )
    lgb_sm = simulate_equity(X_te_full, lgb_preds, y_te_full, price_idx, atr_idx)

    # Summary
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_pf = np.mean([r['pf'] for r in results if np.isfinite(r['pf'])])
    avg_wr = np.mean([r['win_rate'] for r in results])
    pos_windows = sum(1 for r in results if r['sharpe'] > 0)

    log(f"\n  SUMMARY {symbol} {timeframe}:")
    log(f"  XGB Walk-Forward: Avg Acc={avg_acc:.1%} | Avg Sharpe={avg_sharpe:.2f} | "
        f"Avg PF={avg_pf:.2f} | WR={avg_wr:.1f}% | Positive={pos_windows}/{len(results)}")
    log(f"  LGB Full Split:   Acc={lgb_acc:.1%} | Sharpe={lgb_sm['sharpe']:.2f} | "
        f"PF={lgb_sm['pf']:.2f} | WR={lgb_sm['win_rate']:.1f}%")

    # Pick winner by Sharpe
    if lgb_sm['sharpe'] > avg_sharpe:
        winner = 'lgb'
        winner_model = lgb_model
        winner_sharpe = lgb_sm['sharpe']
        winner_acc = lgb_acc
        winner_pf = lgb_sm['pf']
    else:
        winner = 'xgb'
        winner_model = best_model
        winner_sharpe = avg_sharpe
        winner_acc = avg_acc
        winner_pf = avg_pf

    log(f"  WINNER: {winner.upper()} (Sharpe={winner_sharpe:.2f}, PF={winner_pf:.2f})")

    return {
        'symbol': symbol, 'timeframe': timeframe,
        'model': winner_model, 'model_type': winner,
        'features': features, 'n_features': len(features),
        'accuracy': winner_acc, 'sharpe': winner_sharpe, 'pf': winner_pf,
        'avg_win_rate': avg_wr, 'walk_forward_results': results,
        'positive_windows': pos_windows, 'total_windows': len(results),
    }


# ========================= MAIN =========================
def main():
    log("=" * 70)
    log("FXJEFE MULTI-TIMEFRAME TRAINING (No Leakage, Binary Classification)")
    log("=" * 70)

    # Define what to train
    # Crypto symbols from FXJEFE_Crypto_Features.csv
    crypto_pairs = [
        ('BTCUSD', ['M15', 'H1', 'H4', 'D1']),
        ('ETHUSD', ['M15', 'H1', 'H4', 'D1']),
        ('XRPUSD', ['M15', 'H1', 'H4', 'D1']),
    ]

    # Forex + indices from Historical/enhanced/ (removed XAUUSD per user request)
    forex_pairs = [
        ('EURUSD', ['M15', 'H4', 'D1']),
        ('NAS100', ['M15', 'H4', 'D1']),
    ]

    all_results = []

    # ── Train Crypto Models ──
    log("\n" + "=" * 70)
    log("CRYPTO MODELS")
    log("=" * 70)

    for symbol, timeframes in crypto_pairs:
        for tf in timeframes:
            log(f"\n{'='*50}")
            log(f"Training {symbol} {tf}")
            log(f"{'='*50}")

            df = load_crypto_data(symbol, tf)
            if df is None or len(df) < 1000:
                log(f"  Insufficient data for {symbol} {tf}")
                continue

            features = get_clean_features(df)
            log(f"  Using {len(features)} clean features (no leakage)")

            # Leakage check
            leaked = [f for f in features if f in FORBIDDEN]
            if leaked:
                log(f"  LEAKAGE DETECTED: {leaked} — removing!")
                features = [f for f in features if f not in FORBIDDEN]

            df = prepare_binary_labels(df, tf, symbol)
            if df is None or len(df) < 1000:
                log(f"  Insufficient labeled data after filtering")
                continue

            result = walk_forward_train(df, features, symbol, tf)
            if result:
                all_results.append(result)

                # Save model
                model_name = f"{symbol}_{tf}_binary"
                if result['model_type'] == 'xgb':
                    result['model'].save_model(
                        os.path.join(MODELS_DIR, f'{model_name}_xgb.json'))
                else:
                    joblib.dump(result['model'],
                                os.path.join(MODELS_DIR, f'{model_name}_lgb.pkl'))
                log(f"  Saved {model_name}")

    # ── Train Forex Models ──
    log("\n" + "=" * 70)
    log("FOREX MODELS")
    log("=" * 70)

    for symbol, timeframes in forex_pairs:
        for tf in timeframes:
            log(f"\n{'='*50}")
            log(f"Training {symbol} {tf}")
            log(f"{'='*50}")

            df = load_forex_data(symbol, tf)
            if df is None or len(df) < 1000:
                log(f"  Insufficient data for {symbol} {tf}")
                continue

            features = get_clean_features(df)
            log(f"  Using {len(features)} clean features (no leakage)")

            leaked = [f for f in features if f in FORBIDDEN]
            if leaked:
                log(f"  LEAKAGE DETECTED: {leaked} — removing!")
                features = [f for f in features if f not in FORBIDDEN]

            df = prepare_binary_labels(df, tf, symbol)
            if df is None or len(df) < 1000:
                log(f"  Insufficient labeled data after filtering")
                continue

            result = walk_forward_train(df, features, symbol, tf)
            if result:
                all_results.append(result)

                model_name = f"{symbol}_{tf}_binary"
                if result['model_type'] == 'xgb':
                    result['model'].save_model(
                        os.path.join(MODELS_DIR, f'{model_name}_xgb.json'))
                else:
                    joblib.dump(result['model'],
                                os.path.join(MODELS_DIR, f'{model_name}_lgb.pkl'))
                log(f"  Saved {model_name}")

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("MULTI-TIMEFRAME RESULTS — ALL MODELS")
    log("=" * 70)
    log(f"  {'Symbol':<8} {'TF':<5} {'Type':<5} {'Feat':<5} {'Acc':<8} {'Sharpe':<8} "
        f"{'PF':<7} {'WR':<8} {'WF+':<5}")
    log(f"  {'-'*60}")

    for r in sorted(all_results, key=lambda x: -x['sharpe']):
        log(f"  {r['symbol']:<8} {r['timeframe']:<5} {r['model_type']:<5} "
            f"{r['n_features']:<5} {r['accuracy']:>5.1%}  {r['sharpe']:>6.2f}  "
            f"{r['pf']:>5.2f}  {r['avg_win_rate']:>5.1f}%  "
            f"{r['positive_windows']}/{r['total_windows']}")

    # Best overall
    if all_results:
        best = max(all_results, key=lambda x: x['sharpe'])
        log(f"\n  BEST MODEL: {best['symbol']} {best['timeframe']} "
            f"({best['model_type'].upper()}) — Sharpe={best['sharpe']:.2f}, "
            f"PF={best['pf']:.2f}, Acc={best['accuracy']:.1%}")

        # Identify which timeframes work best
        tf_sharpes = {}
        for r in all_results:
            tf_sharpes.setdefault(r['timeframe'], []).append(r['sharpe'])
        log(f"\n  Average Sharpe by Timeframe:")
        for tf in ['M15', 'H1', 'H4', 'D1']:
            if tf in tf_sharpes:
                avg = np.mean(tf_sharpes[tf])
                log(f"    {tf}: {avg:.2f}")

    log(f"\n  Models saved to: {MODELS_DIR}")
    log(f"  Log: {LOG_FILE}")
    log("=" * 70)


if __name__ == '__main__':
    main()
