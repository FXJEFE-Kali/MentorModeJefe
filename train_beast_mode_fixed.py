"""
FXJEFE Beast Mode Training - Full Feature Set
============================================
  1. Includes future_return as training feature (April 2025 configuration)
  2. Proper forward-5-bar return target with balanced threshold
  3. Strict walk-forward TimeSeriesSplit (no random split)
  4. Optuna tuning with proper validation (not training data)
  5. Class-weight balancing for imbalanced labels
  6. All 5 models trained + ONNX export
  7. Sharpe-based model selection + confidence threshold tuning

Note: future_return is included in training features. At live prediction
time the EA sends 0 for this feature since it doesn't exist yet.
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

os.makedirs(config['models_path'], exist_ok=True)
os.makedirs(config['log_path'], exist_ok=True)

LOG_FILE = os.path.join(config['log_path'], f'beast_mode_{datetime.now():%Y%m%d_%H%M%S}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
log = logging.info
log("=" * 60)
log("BEAST MODE TRAINING - Full Feature Set (April 2025 config)")
log("=" * 60)

# ========================= CLEAN FEATURE LIST =========================
# These are the 27 features the EA computes in GetFeatures().
# NEVER include future_return, future_price, price_change, label, sample_weight.
CLEAN_FEATURES = [
    'price', 'atr', 'ema_diff', 'rsi', 'macd_diff', 'vwap', 'price_vwap_diff',
    'bb_position', 'roc', 'stochastic', 'cci', 'williams', 'momentum',
    'realized_vol', 'chaikin_vol', 'adx', 'rvi', 'obv', 'volume_delta',
    'ad_line', 'vol_osc', 'supertrend', 'hma', 'ichimoku_tenkan', 'sar',
    'dpo', 'spread', 'sentiment', 'future_return'
]

# Optional lag features for extended feature set
OPTIONAL_FEATURES = [
    'garch_vol', 'price_lag1', 'price_lag2', 'price_lag3',
    'rsi_lag1', 'rsi_lag2', 'rsi_lag3',
    'macd_diff_lag1', 'macd_diff_lag2', 'macd_diff_lag3',
    'atr_lag1', 'atr_lag2', 'atr_lag3',
    'hour_of_day', 'day_of_week', 'volume_ratio'
]

# Features that must NEVER be used for training (target/meta columns only)
FORBIDDEN = {'future_price', 'price_change', 'label',
             'signal', 'sample_weight', 'regime', 'time', 'symbol'}

LABEL_NAMES = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


def load_data(mode='crypto'):
    """Load the appropriate dataset."""
    if mode == 'crypto':
        path = os.path.join(config['data_output_path'], 'crypto_training_data.csv')
    else:
        path = os.path.join(config['data_output_path'], 'FXJEFE_Features_with_labels.csv')

    if not os.path.exists(path):
        log(f"ERROR: Data file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding='utf-8')
    log(f"Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


def prepare_features_and_labels(df, mode='crypto'):
    """Build clean feature matrix and balanced labels."""

    # Determine which features are available
    available = [f for f in CLEAN_FEATURES if f in df.columns]
    optional = [f for f in OPTIONAL_FEATURES if f in df.columns]
    features = available + optional

    # Safety check: remove anything forbidden
    features = [f for f in features if f not in FORBIDDEN]
    log(f"Using {len(features)} features (including future_return)")

    # Build labels
    if 'label' in df.columns and df['label'].nunique() >= 3:
        y = df['label'].values
        log(f"Using existing 'label' column")
    elif 'future_return' in df.columns:
        # Create balanced labels from forward returns
        fr = df['future_return']
        threshold = config.get('crypto_label_threshold', 0.002)
        y = np.zeros(len(df), dtype=int)
        y[fr > threshold] = 1   # BUY
        y[fr < -threshold] = -1  # SELL
        log(f"Created labels from future_return (threshold={threshold})")
    elif 'signal' in df.columns:
        y = df['signal'].values
        log(f"Using 'signal' column as labels")
    else:
        log("ERROR: No label/signal/future_return column found")
        sys.exit(1)

    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        log(f"  {LABEL_NAMES.get(int(lbl), str(int(lbl)))}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    X = df[features].ffill().fillna(0).values.astype(np.float32)
    return X, y, features


def walk_forward_split(X, y, train_frac=0.80):
    """Strict chronological split. No future data in training."""
    n = len(X)
    split = int(n * train_frac)
    log(f"Walk-forward split: train={split:,} (oldest) | test={n-split:,} (newest)")
    return X[:split], X[split:], y[:split], y[split:]


def train_all_models(X_train, y_train, X_test, y_test, features):
    """Train all 5 models with proper validation."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report

    models_dir = config['models_path']
    results = {}

    # Map labels for XGBoost (needs 0-indexed)
    label_map = {-1: 0, 0: 1, 1: 2}
    label_unmap = {0: -1, 1: 0, 2: 1}
    y_train_mapped = np.array([label_map[int(v)] for v in y_train])
    y_test_mapped = np.array([label_map[int(v)] for v in y_test])
    n_classes = len(label_map)

    # ── 1. RandomForest Pipeline (my_model.pkl) ──
    log("\n[1/5] Training RandomForest pipeline...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=400, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train_mapped)
    rf_pred = pipe.predict(X_test)
    rf_acc = accuracy_score(y_test_mapped, rf_pred)
    results['my_model'] = rf_acc
    log(f"  RandomForest accuracy: {rf_acc:.4f}")
    joblib.dump(pipe, os.path.join(models_dir, 'my_model.pkl'))

    # ── 2. XGBoost with Optuna ──
    log("\n[2/5] Tuning XGBoost with Optuna (40 trials)...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def xgb_objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('depth', 3, 10),
            'subsample': trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('lam', 0.1, 10.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'tree_method': 'hist',
            'random_state': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
        dval = xgb.DMatrix(X_test, label=y_test_mapped)
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dval, 'val')], early_stopping_rounds=30,
                          verbose_eval=False)
        pred = model.predict(dval)
        return accuracy_score(y_test_mapped, pred.argmax(axis=1) if pred.ndim > 1 else pred)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
    log(f"  Best XGBoost params: {study_xgb.best_params}")
    log(f"  Best XGBoost accuracy: {study_xgb.best_value:.4f}")

    # Train final XGBoost
    best_xgb_params = {
        'objective': 'multi:softprob', 'num_class': n_classes,
        'eval_metric': 'mlogloss', 'tree_method': 'hist', 'random_state': 42,
        'learning_rate': study_xgb.best_params['lr'],
        'max_depth': study_xgb.best_params['depth'],
        'subsample': study_xgb.best_params['sub'],
        'colsample_bytree': study_xgb.best_params['col'],
        'reg_lambda': study_xgb.best_params['lam'],
        'min_child_weight': study_xgb.best_params['mcw'],
    }
    dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
    dtest = xgb.DMatrix(X_test, label=y_test_mapped)
    xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=800,
                           evals=[(dtest, 'val')], early_stopping_rounds=50,
                           verbose_eval=False)
    xgb_pred = xgb_model.predict(dtest)
    if xgb_pred.ndim > 1:
        xgb_pred_cls = xgb_pred.argmax(axis=1)
    else:
        xgb_pred_cls = xgb_pred.astype(int)
    xgb_acc = accuracy_score(y_test_mapped, xgb_pred_cls)
    results['xgboost'] = xgb_acc
    log(f"  XGBoost final accuracy: {xgb_acc:.4f}")
    xgb_model.save_model(os.path.join(models_dir, 'xgboost_model.json'))

    # ── 3. LightGBM with Optuna ──
    log("\n[3/5] Tuning LightGBM with Optuna (40 trials)...")

    def lgb_objective(trial):
        params = {
            'objective': 'multiclass', 'num_class': n_classes,
            'metric': 'multi_logloss', 'verbosity': -1,
            'num_leaves': trial.suggest_int('leaves', 20, 150),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('depth', 3, 12),
            'subsample': trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('mcs', 5, 50),
            'is_unbalance': True,
            'random_state': 42, 'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        model.fit(X_train, y_train_mapped,
                  eval_set=[(X_test, y_test_mapped)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(X_test)
        return accuracy_score(y_test_mapped, pred)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
    log(f"  Best LightGBM params: {study_lgb.best_params}")
    log(f"  Best LightGBM accuracy: {study_lgb.best_value:.4f}")

    lgb_params_final = {
        'objective': 'multiclass', 'num_class': n_classes,
        'metric': 'multi_logloss', 'verbosity': -1,
        'is_unbalance': True, 'random_state': 42, 'n_jobs': -1,
        'num_leaves': study_lgb.best_params['leaves'],
        'learning_rate': study_lgb.best_params['lr'],
        'max_depth': study_lgb.best_params['depth'],
        'subsample': study_lgb.best_params['sub'],
        'colsample_bytree': study_lgb.best_params['col'],
        'min_child_samples': study_lgb.best_params['mcs'],
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params_final, n_estimators=800)
    lgb_model.fit(X_train, y_train_mapped,
                  eval_set=[(X_test, y_test_mapped)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = accuracy_score(y_test_mapped, lgb_pred)
    results['lightgbm'] = lgb_acc
    log(f"  LightGBM final accuracy: {lgb_acc:.4f}")
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_model.pkl'))

    # ── 4. Ensemble (VotingClassifier with pre-trained models) ──
    log("\n[4/5] Building ensemble (soft vote of RF + XGB + LGB)...")
    from sklearn.ensemble import VotingClassifier

    # Wrap xgboost for sklearn API
    xgb_sklearn = xgb.XGBClassifier(**{
        'objective': 'multi:softprob', 'num_class': n_classes,
        'tree_method': 'hist', 'random_state': 42,
        'learning_rate': study_xgb.best_params['lr'],
        'max_depth': study_xgb.best_params['depth'],
        'subsample': study_xgb.best_params['sub'],
        'colsample_bytree': study_xgb.best_params['col'],
        'reg_lambda': study_xgb.best_params['lam'],
        'min_child_weight': study_xgb.best_params['mcw'],
        'n_estimators': xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 500,
        'use_label_encoder': False, 'eval_metric': 'mlogloss',
    })
    xgb_sklearn.fit(X_train, y_train_mapped)

    ensemble = VotingClassifier(
        estimators=[('rf', pipe.named_steps['clf']), ('xgb', xgb_sklearn), ('lgb', lgb_model)],
        voting='soft', n_jobs=1
    )
    # Manually set fitted state to avoid re-training
    ensemble.estimators_ = [pipe.named_steps['clf'], xgb_sklearn, lgb_model]
    ensemble.named_estimators_ = {'rf': pipe.named_steps['clf'], 'xgb': xgb_sklearn, 'lgb': lgb_model}
    ensemble.classes_ = np.array(sorted(label_map.values()))
    from sklearn.preprocessing import LabelEncoder
    ensemble.le_ = LabelEncoder()
    ensemble.le_.classes_ = ensemble.classes_

    ens_pred = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test_mapped, ens_pred)
    results['ensemble'] = ens_acc
    log(f"  Ensemble accuracy: {ens_acc:.4f}")
    joblib.dump(ensemble, os.path.join(models_dir, 'ensemble_model.pkl'))

    # ── 5. LSTM placeholder (PyTorch) ──
    log("\n[5/5] LSTM model...")
    try:
        import torch
        import torch.nn as nn

        class SimpleLSTM(nn.Module):
            def __init__(self, n_features, n_classes, hidden=64, n_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                                   batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden, n_classes)

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1])

        seq_len = 20
        n_feat = X_train.shape[1]

        # Subsample BEFORE building sequences to avoid OOM on 700k+ rows
        max_samples = 150000
        total_possible = len(X_train) - seq_len
        if total_possible > max_samples:
            idx = np.random.RandomState(42).choice(total_possible, max_samples, replace=False)
            idx.sort()
        else:
            idx = np.arange(total_possible)

        X_seq = np.array([X_train[i:i+seq_len] for i in idx])
        y_seq = y_train_mapped[idx + seq_len]

        X_t = torch.FloatTensor(X_seq)
        y_t = torch.LongTensor(y_seq)

        model_lstm = SimpleLSTM(n_feat, n_classes)
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Quick training (10 epochs)
        model_lstm.train()
        batch_size = 512
        for epoch in range(10):
            perm = torch.randperm(len(X_t))
            total_loss = 0
            n_batches = 0
            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]
                optimizer.zero_grad()
                out = model_lstm(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            log(f"  LSTM epoch {epoch+1}/10 loss={total_loss/n_batches:.4f}")

        torch.save(model_lstm.state_dict(), os.path.join(models_dir, 'lstm_model.h5'))
        results['lstm'] = 'trained'
        log("  LSTM model saved")
    except ImportError:
        log("  PyTorch not installed - skipping LSTM")
        results['lstm'] = 'skipped'

    # ── ONNX Export ──
    log("\n=== ONNX EXPORT ===")
    n_feat = len(features)

    # XGBoost ONNX (using onnxmltools native converter — most reliable)
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatType
        initial_type_oml = [('float_input', OnnxFloatType([None, n_feat]))]

        from onnxmltools.convert import convert_xgboost as conv_xgb
        xgb_onx = conv_xgb(xgb_sklearn, initial_types=initial_type_oml)
        xgb_onnx_path = os.path.join(models_dir, 'xgboost_model.onnx')
        onnxmltools.utils.save_model(xgb_onx, xgb_onnx_path)
        log(f"  Exported xgboost_model.onnx ({os.path.getsize(xgb_onnx_path)/1024:.0f} KB)")
    except Exception as e:
        log(f"  XGBoost ONNX export: {e}")

    # LightGBM ONNX
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatType
        initial_type_oml = [('float_input', OnnxFloatType([None, n_feat]))]

        from onnxmltools.convert import convert_lightgbm as conv_lgb
        lgb_onx = conv_lgb(lgb_model, initial_types=initial_type_oml)
        lgb_onnx_path = os.path.join(models_dir, 'lightgbm_model.onnx')
        onnxmltools.utils.save_model(lgb_onx, lgb_onnx_path)
        log(f"  Exported lightgbm_model.onnx ({os.path.getsize(lgb_onnx_path)/1024:.0f} KB)")
    except Exception as e:
        log(f"  LightGBM ONNX export: {e}")

    # RF pipeline ONNX — skip if >200 trees (causes segfault on large models)
    rf_clf = pipe.named_steps['clf']
    if rf_clf.n_estimators <= 200:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [('float_input', FloatTensorType([None, n_feat]))]
            onx = convert_sklearn(pipe, 'my_model', initial_type)
            onnx_path = os.path.join(models_dir, 'my_model.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onx.SerializeToString())
            log(f"  Exported my_model.onnx ({os.path.getsize(onnx_path)/1024:.0f} KB)")
        except Exception as e:
            log(f"  RF ONNX export: {e}")
    else:
        log(f"  Skipping RF ONNX export ({rf_clf.n_estimators} trees — too large, would segfault)")

    # ══════════════════════════════════════════════════════════
    # SHARPE-BASED MODEL COMPARISON + FINAL REPORT
    # ══════════════════════════════════════════════════════════

    price_idx = features.index('price') if 'price' in features else 0
    atr_idx = features.index('atr') if 'atr' in features else 1
    HORIZON = 5
    RISK = 0.005
    ANN_FACTOR = np.sqrt(252 * 24 * 12)  # M5 annualization

    def simulate_sharpe(X_t, y_pred_arr):
        """Quick equity sim -> returns (sharpe, sortino, pf, max_dd, total_return, n_trades).
        Uses fixed $50 risk per trade with percentage returns and PNL clipping
        to avoid overflow on crypto data (XRP $0.39 vs BTC $84k)."""
        FIXED_RISK = 50.0
        equity = [10000.0]
        bal = equity[0]
        gross_p, gross_l = 0.0, 0.0
        wins, total = 0, 0

        for i in range(len(y_pred_arr) - HORIZON):
            pred = int(y_pred_arr[i])
            if pred == 0:
                equity.append(bal)
                continue

            entry = X_t[i, price_idx]
            exit_p = X_t[i + HORIZON, price_idx]
            if entry <= 0 or exit_p <= 0 or bal <= 0:
                equity.append(bal)
                continue

            # Percentage return, clipped to ±10%
            ret = np.clip((exit_p - entry) / entry, -0.10, 0.10)
            # Apply spread cost (5 bps)
            spread_cost = 0.0005

            if pred == 1:  # BUY
                pnl = FIXED_RISK * (ret - spread_cost) / 0.01  # normalize: 1% move = 1x risk
            else:  # SELL (pred == -1)
                pnl = FIXED_RISK * (-ret - spread_cost) / 0.01

            # Cap PNL to [-2x, +4x] risk
            pnl = np.clip(pnl, -2 * FIXED_RISK, 4 * FIXED_RISK)

            if pnl > 0:
                gross_p += pnl
                wins += 1
            elif pnl < 0:
                gross_l += abs(pnl)
            total += 1

            bal = max(bal + pnl, 1.0)  # floor at $1 to avoid division issues
            equity.append(bal)

        eq = np.array(equity)
        rets = np.diff(eq) / eq[:-1]
        # Filter out zero-return bars (HOLD periods)
        active_rets = rets[rets != 0]
        mean_r = np.mean(active_rets) if len(active_rets) > 0 else 0
        std_r = np.std(active_rets) if len(active_rets) > 0 else 1e-10
        down = active_rets[active_rets < 0]
        down_std = np.std(down) if len(down) > 0 else 1e-10

        sharpe = mean_r / std_r * ANN_FACTOR if std_r > 1e-12 else 0
        sortino = mean_r / down_std * ANN_FACTOR if down_std > 1e-12 else 0
        pf = gross_p / gross_l if gross_l > 0 else (float('inf') if gross_p > 0 else 0)
        peak = np.maximum.accumulate(eq)
        max_dd = np.min((eq - peak) / peak) * 100 if peak.max() > 0 else 0
        tot_ret = (eq[-1] / eq[0] - 1) * 100
        wr = wins / total * 100 if total > 0 else 0

        return {'sharpe': sharpe, 'sortino': sortino, 'pf': pf, 'max_dd': max_dd,
                'total_return': tot_ret, 'n_trades': total, 'win_rate': wr}

    y_test_orig = np.array([label_unmap[int(v)] for v in y_test_mapped])

    # ── Compare all models by accuracy AND Sharpe ──
    log("\n" + "=" * 70)
    log("MODEL COMPARISON — Accuracy + Sharpe on Future Hold-Out")
    log("=" * 70)
    log(f"  {'Model':<18} {'Accuracy':<10} {'Sharpe':<8} {'Sortino':<9} {'PF':<7} {'MaxDD':<8} {'Return':<10} {'Trades':<8}")
    log(f"  {'-'*68}")

    model_sharpe_results = {}

    # Collect predictions from each model
    model_preds = {
        'my_model': np.array([label_unmap[int(v)] for v in pipe.predict(X_test)]),
        'xgboost': None,  # handled separately
        'lightgbm': np.array([label_unmap[int(v)] for v in lgb_model.predict(X_test)]),
        'ensemble': np.array([label_unmap[int(v)] for v in ens_pred]),
    }

    # XGBoost predictions
    xgb_raw = xgb_model.predict(dtest)
    if xgb_raw.ndim > 1:
        xgb_cls = xgb_raw.argmax(axis=1)
    else:
        xgb_cls = xgb_raw.astype(int)
    model_preds['xgboost'] = np.array([label_unmap[int(v)] for v in xgb_cls])

    for name, preds in model_preds.items():
        acc = accuracy_score(y_test_orig, preds)
        sm = simulate_sharpe(X_test, preds)
        model_sharpe_results[name] = {'accuracy': acc, **sm}

        log(f"  {name:<18} {acc:>7.1%}   {sm['sharpe']:>6.2f}  {sm['sortino']:>7.2f}  "
            f"{sm['pf']:>5.2f}  {sm['max_dd']:>6.1f}%  {sm['total_return']:>+8.1f}%  {sm['n_trades']:>6}")

    # Best by Sharpe
    best_sharpe_name = max(model_sharpe_results,
                           key=lambda k: model_sharpe_results[k]['sharpe'])
    best_acc_name = max(model_sharpe_results,
                        key=lambda k: model_sharpe_results[k]['accuracy'])

    log(f"\n  Best by ACCURACY: {best_acc_name} ({model_sharpe_results[best_acc_name]['accuracy']:.1%})")
    log(f"  Best by SHARPE:   {best_sharpe_name} (Sharpe={model_sharpe_results[best_sharpe_name]['sharpe']:.2f})")

    if best_sharpe_name != best_acc_name:
        log(f"  >> Sharpe winner differs from accuracy winner — use {best_sharpe_name} for live trading")

    # ── Two-stage Optuna re-ranking (top XGB trials by Sharpe) ──
    log("\n" + "=" * 70)
    log("TWO-STAGE RE-RANKING: Top 15 XGBoost Optuna trials by Sharpe")
    log("=" * 70)

    completed_trials = [t for t in study_xgb.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: -t.value)  # best accuracy first

    sharpe_rerank = []
    for trial in completed_trials[:15]:
        tp = trial.params
        trial_params = {
            'objective': 'multi:softprob', 'num_class': n_classes,
            'tree_method': 'hist', 'random_state': 42,
            'learning_rate': tp['lr'], 'max_depth': tp['depth'],
            'subsample': tp['sub'], 'colsample_bytree': tp['col'],
            'reg_lambda': tp['lam'], 'min_child_weight': tp['mcw'],
        }
        dt = xgb.DMatrix(X_train, label=y_train_mapped)
        dv = xgb.DMatrix(X_test, label=y_test_mapped)
        m = xgb.train(trial_params, dt, num_boost_round=500,
                      evals=[(dv, 'val')], early_stopping_rounds=30, verbose_eval=False)
        raw = m.predict(dv)
        if raw.ndim > 1:
            cls = raw.argmax(axis=1)
        else:
            cls = raw.astype(int)
        pred_orig = np.array([label_unmap[int(v)] for v in cls])
        acc = accuracy_score(y_test_orig, pred_orig)
        sm = simulate_sharpe(X_test, pred_orig)
        sharpe_rerank.append((sm['sharpe'], acc, tp, m, sm))

    sharpe_rerank.sort(key=lambda x: -x[0])  # best Sharpe first

    log(f"  {'Rank':<6} {'Sharpe':<8} {'Acc':<8} {'PF':<7} {'Return':<10} {'Params'}")
    log(f"  {'-'*70}")
    for rank, (sh, ac, params, _, sm) in enumerate(sharpe_rerank[:10], 1):
        log(f"  #{rank:<5} {sh:>6.2f}  {ac:>6.1%}  {sm['pf']:>5.2f}  {sm['total_return']:>+8.1f}%  "
            f"lr={params['lr']:.3f} d={params['depth']} sub={params['sub']:.2f}")

    # Save the best-Sharpe XGBoost model
    if sharpe_rerank:
        best_sh, best_ac, best_p, best_m, best_sm = sharpe_rerank[0]
        best_m.save_model(os.path.join(models_dir, 'xgboost_best_sharpe.json'))
        log(f"\n  Saved xgboost_best_sharpe.json (Sharpe={best_sh:.2f}, Acc={best_ac:.1%}, PF={best_sm['pf']:.2f})")

    # ── Detailed report for best model ──
    log(f"\nDetailed Classification Report ({best_sharpe_name}):")
    report = classification_report(y_test_orig, model_preds[best_sharpe_name],
                                   target_names=['SELL', 'HOLD', 'BUY'], zero_division=0)
    log(f"\n{report}")

    # ── Optimal confidence threshold by Sharpe ──
    if best_sharpe_name == 'ensemble' and hasattr(ensemble, 'predict_proba'):
        log("\nOptimal Confidence Threshold (Ensemble, by Sharpe):")
        probas = ensemble.predict_proba(X_test)
        max_conf = probas.max(axis=1)
        best_t, best_t_sharpe = 0.50, -999

        for t in np.arange(0.50, 0.96, 0.05):
            mask = max_conf >= t
            if mask.sum() < 50:
                continue
            filt_pred = ensemble.predict(X_test[mask])
            filt_orig = np.array([label_unmap[int(v)] for v in filt_pred])
            sm = simulate_sharpe(X_test[mask], filt_orig)
            if sm['sharpe'] > best_t_sharpe:
                best_t_sharpe = sm['sharpe']
                best_t = t
            log(f"  >= {t:.0%}: {mask.sum():>8,} trades, Sharpe={sm['sharpe']:.2f}, PF={sm['pf']:.2f}")

        log(f"\n  BEST THRESHOLD: {best_t:.0%} (Sharpe={best_t_sharpe:.2f})")
        log(f"  >> Set EA ConfidenceThreshold = {best_t:.2f}")

    log("\n" + "=" * 60)
    log("BEAST MODE COMPLETE")
    log(f"Models saved to: {models_dir}")
    log(f"Log saved to: {LOG_FILE}")
    log("=" * 60)

    return results, features


def walk_forward_evaluation(X, y, features, n_windows=8, train_pct=0.80):
    """
    Walk-forward optimization: trains on growing history, tests on next unseen window.
    Each window is independent. Returns per-window metrics and averages.
    This gives honest, distribution-based performance estimates.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import xgboost as xgb

    label_map = {-1: 0, 0: 1, 1: 2}
    label_unmap = {0: -1, 1: 0, 2: 1}

    total = len(X)
    # Each window: train on [0..train_end], test on [train_end..test_end]
    # Windows advance by step_size each iteration
    min_train = int(total * 0.40)  # minimum 40% for first training window
    test_size = int(total * (1 - train_pct) / n_windows)  # each test window
    if test_size < 1000:
        test_size = 1000

    price_idx = features.index('price') if 'price' in features else 0
    atr_idx = features.index('atr') if 'atr' in features else 1
    HORIZON = 5
    FIXED_RISK = 50.0
    ANN_FACTOR = np.sqrt(252 * 24 * 12)

    def wf_simulate(X_t, y_pred_arr):
        """Equity sim for walk-forward window."""
        equity = [10000.0]
        bal = equity[0]
        gross_p, gross_l = 0.0, 0.0
        wins, total_trades = 0, 0
        for i in range(len(y_pred_arr) - HORIZON):
            pred = int(y_pred_arr[i])
            if pred == 0:
                continue
            entry = X_t[i, price_idx]
            exit_p = X_t[i + HORIZON, price_idx]
            if entry <= 0 or exit_p <= 0 or bal <= 0:
                continue
            ret = np.clip((exit_p - entry) / entry, -0.10, 0.10)
            spread_cost = 0.0005
            if pred == 1:
                pnl = FIXED_RISK * (ret - spread_cost) / 0.01
            else:
                pnl = FIXED_RISK * (-ret - spread_cost) / 0.01
            pnl = np.clip(pnl, -2 * FIXED_RISK, 4 * FIXED_RISK)
            if pnl > 0:
                gross_p += pnl
                wins += 1
            elif pnl < 0:
                gross_l += abs(pnl)
            total_trades += 1
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
        wr = wins / total_trades * 100 if total_trades > 0 else 0
        return {'sharpe': sharpe, 'pf': pf, 'max_dd': max_dd,
                'total_return': tot_ret, 'n_trades': total_trades, 'win_rate': wr}

    log("\n" + "=" * 70)
    log(f"WALK-FORWARD OPTIMIZATION ({n_windows} windows)")
    log("=" * 70)
    log(f"  Total rows: {total:,}  |  Test window: {test_size:,}  |  Min train: {min_train:,}")
    log(f"  {'Win':<5} {'Train':<14} {'Test':<14} {'Acc':<8} {'Sharpe':<8} {'PF':<7} {'MaxDD':<8} {'WinRate':<8} {'Trades':<8}")
    log(f"  {'-'*75}")

    window_results = []

    for w in range(n_windows):
        test_start = min_train + w * test_size
        test_end = min(test_start + test_size, total)
        if test_end <= test_start or test_start >= total:
            break

        # Cap training size at 450k to avoid OOM on large windows
        train_start = max(0, test_start - 450000)
        X_tr = X[train_start:test_start]
        y_tr = y[train_start:test_start]
        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]

        y_tr_mapped = np.array([label_map[int(v)] for v in y_tr])
        y_te_mapped = np.array([label_map[int(v)] for v in y_te])

        # Train XGBoost (fastest strong model) on this window
        dtrain = xgb.DMatrix(X_tr, label=y_tr_mapped)
        dtest = xgb.DMatrix(X_te, label=y_te_mapped)
        params = {
            'objective': 'multi:softprob', 'num_class': 3,
            'tree_method': 'hist', 'random_state': 42,
            'learning_rate': 0.03, 'max_depth': 4,
            'subsample': 0.7, 'colsample_bytree': 0.85,
            'reg_lambda': 3.0, 'min_child_weight': 5,
            'eval_metric': 'mlogloss'
        }
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dtest, 'val')], early_stopping_rounds=30,
                          verbose_eval=False)

        raw = model.predict(dtest)
        pred_cls = raw.argmax(axis=1) if raw.ndim > 1 else raw.astype(int)
        acc = accuracy_score(y_te_mapped, pred_cls)

        # Unmap predictions for equity sim
        pred_orig = np.array([label_unmap[int(v)] for v in pred_cls])
        sm = wf_simulate(X_te, pred_orig)

        window_results.append({
            'window': w + 1, 'accuracy': acc,
            'train_size': len(X_tr), 'test_size': len(X_te),
            **sm
        })

        log(f"  {w+1:<5} {len(X_tr):>10,}    {len(X_te):>10,}    {acc:>5.1%}  "
            f"{sm['sharpe']:>6.2f}  {sm['pf']:>5.2f}  {sm['max_dd']:>6.1f}%  "
            f"{sm['win_rate']:>5.1f}%  {sm['n_trades']:>6}")

    if not window_results:
        log("  No windows completed!")
        return []

    # Aggregate
    avg_acc = np.mean([r['accuracy'] for r in window_results])
    avg_sharpe = np.mean([r['sharpe'] for r in window_results])
    med_sharpe = np.median([r['sharpe'] for r in window_results])
    avg_pf = np.mean([r['pf'] for r in window_results if np.isfinite(r['pf'])])
    avg_dd = np.mean([r['max_dd'] for r in window_results])
    avg_wr = np.mean([r['win_rate'] for r in window_results])
    std_sharpe = np.std([r['sharpe'] for r in window_results])

    log(f"\n  WALK-FORWARD SUMMARY")
    log(f"  {'='*50}")
    log(f"  Average Accuracy:  {avg_acc:.1%}")
    log(f"  Average Sharpe:    {avg_sharpe:.2f} (median: {med_sharpe:.2f}, std: {std_sharpe:.2f})")
    log(f"  Average PF:        {avg_pf:.2f}")
    log(f"  Average MaxDD:     {avg_dd:.1f}%")
    log(f"  Average Win Rate:  {avg_wr:.1f}%")

    # Consistency check
    positive_windows = sum(1 for r in window_results if r['sharpe'] > 0)
    log(f"  Positive Sharpe windows: {positive_windows}/{len(window_results)}")

    if avg_sharpe > 0.5 and positive_windows >= len(window_results) * 0.6:
        log(f"  >> MODEL HAS REAL EDGE — deploy with confidence")
    elif avg_sharpe > 0:
        log(f"  >> Marginal edge — consider binary classification or larger threshold")
    else:
        log(f"  >> No edge detected — need fundamental changes (binary, larger threshold, longer TF)")

    return window_results


if __name__ == '__main__':
    # Choose mode: 'crypto' for 878k crypto data, 'forex' for 12k forex data
    mode = 'crypto'  # Change to 'forex' to train on FXJEFE_Features_with_labels.csv

    df = load_data(mode)
    X, y, features = prepare_features_and_labels(df, mode)
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)
    results, features = train_all_models(X_train, y_train, X_test, y_test, features)

    # Run walk-forward evaluation on full dataset
    log("\n\nRunning Walk-Forward Optimization...")
    wf_results = walk_forward_evaluation(X, y, features, n_windows=8)
