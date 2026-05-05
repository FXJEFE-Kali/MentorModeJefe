"""
FXJEFE Comprehensive Model Accuracy Test
==========================================
Tests ALL saved binary models (XGBoost + LightGBM) against their training data
using walk-forward test splits (last 20%).

Reports:
  - Classification: accuracy, precision, recall, F1, confusion matrix
  - Trading: Sharpe ratio, profit factor, win rate, max drawdown
  - Confidence: accuracy at different confidence thresholds
  - Feature coverage: how many expected features are available
  - Per-model and aggregate summary
"""
import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score
)

warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- Project setup --
PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

CONFIG_PATH = os.path.join(PROJECT, 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

MODELS_DIR = config['models_path']
DATA_DIR = config['data_output_path']
HIST_DIR = os.path.join(DATA_DIR, 'Historical', 'enhanced')

from full_pipeline import (
    compute_all_features, create_labels, select_features,
    simulate_equity, FORBIDDEN, CRYPTO_SYMBOLS, INDEX_SYMBOLS,
    get_asset_class, THRESHOLDS, HORIZONS
)


# ===============================================================
# DATA LOADING
# ===============================================================

def find_data_file(symbol, timeframe):
    """Find the best data file for a symbol/timeframe combo."""
    tf_map = {'D1': 'Daily', 'W1': 'Weekly'}
    tf_name = tf_map.get(timeframe, timeframe)

    # Try enhanced OHLCV first
    pattern = os.path.join(HIST_DIR, f'enhanced_{symbol}_{tf_name}_*.csv')
    matches = glob.glob(pattern)
    if matches:
        return matches[0], 'enhanced'

    # Try raw Historical
    pattern = os.path.join(DATA_DIR, 'Historical', f'{symbol}_{tf_name}_*.csv')
    matches = [m for m in glob.glob(pattern) if 'enhanced' not in m]
    if matches:
        return matches[0], 'raw'

    # Try crypto features CSV
    crypto_path = os.path.join(DATA_DIR, 'FXJEFE_Crypto_Features.csv')
    if os.path.exists(crypto_path):
        return crypto_path, 'crypto_features'

    return None, None


def load_and_prepare(symbol, timeframe, model_features):
    """Load data, compute features, create labels, return ready-to-test data."""
    path, source = find_data_file(symbol, timeframe)
    if path is None:
        return None, None, None, 0

    if source == 'crypto_features':
        df = pd.read_csv(path)
        df = df[(df['symbol'] == symbol) & (df['timeframe'] == timeframe)].copy()
        df = df.reset_index(drop=True)
        if len(df) == 0:
            return None, None, None, 0
        # These don't have OHLCV, use as-is
        has_ohlcv = False
    else:
        df = pd.read_csv(path)
        has_ohlcv = all(c in df.columns for c in ['open', 'high', 'low', 'close'])

    n_rows = len(df)

    # Compute features if we have OHLCV
    if has_ohlcv:
        df = compute_all_features(df)

    # Create labels
    df_labeled = create_labels(df, symbol, timeframe, mode='binary')
    if df_labeled is None or len(df_labeled) < 200:
        return None, None, None, n_rows

    # Map model features to available columns
    available = set(df_labeled.columns)
    matched = [f for f in model_features if f in available]
    missing = [f for f in model_features if f not in available]
    coverage = len(matched) / len(model_features) * 100

    # Fill missing features with 0
    for f in missing:
        df_labeled[f] = 0.0

    return df_labeled, missing, coverage, n_rows


# ===============================================================
# MODEL LOADING
# ===============================================================

def load_model(base_name, model_type):
    """Load XGBoost or LightGBM model."""
    if model_type == 'xgb':
        import xgboost as xgb
        path = os.path.join(MODELS_DIR, f'{base_name}_xgb.json')
        if not os.path.exists(path):
            return None
        model = xgb.Booster()
        model.load_model(path)
        return model
    else:  # lgb
        path = os.path.join(MODELS_DIR, f'{base_name}_lgb.pkl')
        if not os.path.exists(path):
            return None
        return joblib.load(path)


# ===============================================================
# EVALUATION
# ===============================================================

def evaluate_model(model, model_type, X_test, y_test, features, symbol, timeframe):
    """Full evaluation: classification + trading metrics."""
    import xgboost as xgb

    # Get predictions + probabilities
    if model_type == 'xgb':
        dtest = xgb.DMatrix(X_test, feature_names=features)
        probas = model.predict(dtest)  # P(class=1)
        y_pred = (probas > 0.5).astype(int)
        confidence = np.where(probas > 0.5, probas, 1 - probas)
    else:  # lgb
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            probas_2d = model.predict_proba(X_test)
            probas = probas_2d[:, 1]  # P(class=1)
            confidence = np.max(probas_2d, axis=1)
        else:
            probas = None
            confidence = np.ones(len(y_pred)) * 0.5

    # ── Classification metrics ──
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, probas) if probas is not None else 0
    except:
        auc = 0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # ── Trading simulation ──
    price_idx = features.index('price') if 'price' in features else 0
    horizon = HORIZONS.get(timeframe, 5)
    sim = simulate_equity(X_test, y_pred, price_idx, horizon=horizon)

    # ── Confidence analysis ──
    conf_results = []
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = confidence >= thresh
        n_filtered = mask.sum()
        if n_filtered > 10:
            filtered_acc = accuracy_score(y_test[mask], y_pred[mask])
            conf_results.append({
                'threshold': thresh, 'n_trades': n_filtered,
                'pct': n_filtered / len(y_test) * 100,
                'accuracy': filtered_acc
            })

    # ── Per-class breakdown ──
    buy_mask = y_test == 1
    sell_mask = y_test == 0
    buy_acc = accuracy_score(y_test[buy_mask], y_pred[buy_mask]) if buy_mask.sum() > 0 else 0
    sell_acc = accuracy_score(y_test[sell_mask], y_pred[sell_mask]) if sell_mask.sum() > 0 else 0

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'confusion_matrix': cm,
        'buy_acc': buy_acc, 'sell_acc': sell_acc,
        'buy_count': int(buy_mask.sum()), 'sell_count': int(sell_mask.sum()),
        'pred_buy': int((y_pred == 1).sum()), 'pred_sell': int((y_pred == 0).sum()),
        'sharpe': sim['sharpe'], 'pf': sim['pf'], 'win_rate': sim['win_rate'],
        'max_dd': sim['max_dd'], 'n_trades': sim['n_trades'],
        'confidence_analysis': conf_results,
        'mean_confidence': float(confidence.mean()),
        'test_size': len(y_test),
    }


# ===============================================================
# WALK-FORWARD EVALUATION (multiple windows)
# ===============================================================

def walk_forward_eval(model, model_type, X, y, features, symbol, timeframe, n_windows=5):
    """Evaluate model across multiple walk-forward windows (no retraining)."""
    import xgboost as xgb
    total = len(X)
    min_train = max(int(total * 0.40), 1000)
    test_size = max(int(total * 0.10), 200)
    price_idx = features.index('price') if 'price' in features else 0
    horizon = HORIZONS.get(timeframe, 5)

    window_results = []
    all_y_true = []
    all_y_pred = []
    all_conf = []

    for w in range(n_windows):
        test_start = min_train + w * test_size
        test_end = min(test_start + test_size, total)
        if test_end <= test_start or test_start >= total:
            break

        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]

        if len(np.unique(y_te)) < 2:
            continue

        if model_type == 'xgb':
            dtest = xgb.DMatrix(X_te, feature_names=features)
            probas = model.predict(dtest)
            preds = (probas > 0.5).astype(int)
            conf = np.where(probas > 0.5, probas, 1 - probas)
        else:
            preds = model.predict(X_te)
            if hasattr(model, 'predict_proba'):
                probas_2d = model.predict_proba(X_te)
                conf = np.max(probas_2d, axis=1)
            else:
                conf = np.ones(len(preds)) * 0.5

        acc = accuracy_score(y_te, preds)
        sim = simulate_equity(X_te, preds, price_idx, horizon=horizon)
        window_results.append({'window': w+1, 'acc': acc, **sim, 'n': len(y_te)})
        all_y_true.extend(y_te)
        all_y_pred.extend(preds)
        all_conf.extend(conf)

    return window_results, np.array(all_y_true), np.array(all_y_pred), np.array(all_conf)


# ===============================================================
# MAIN
# ===============================================================

def main():
    print("=" * 80)
    print("  FXJEFE MODEL ACCURACY TEST -- Comprehensive Evaluation")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)

    # Discover all models with feature files
    feature_files = glob.glob(os.path.join(MODELS_DIR, '*_binary_features.json'))
    if not feature_files:
        print("No models found!")
        return

    print(f"\nFound {len(feature_files)} model configurations\n")

    all_results = []
    grand_y_true = []
    grand_y_pred = []

    for feat_path in sorted(feature_files):
        with open(feat_path) as f:
            feat_info = json.load(f)

        symbol = feat_info['symbol']
        timeframe = feat_info['timeframe']
        model_type = feat_info['model_type']
        features = feat_info['features']
        base_name = f"{symbol}_{timeframe}_binary"

        print(f"\n{'-' * 70}")
        print(f"  {symbol} {timeframe} ({model_type.upper()}) -- {len(features)} features")
        print(f"{'-' * 70}")

        # Load model
        model = load_model(base_name, model_type)
        if model is None:
            print(f"  X Model file not found -- SKIPPED")
            continue

        # Load & prepare data
        df, missing_feats, coverage, n_rows = load_and_prepare(symbol, timeframe, features)
        if df is None:
            print(f"  X No data available ({n_rows} raw rows) -- SKIPPED")
            continue

        if missing_feats:
            print(f"  Feature coverage: {coverage:.0f}% ({len(features) - len(missing_feats)}/{len(features)})")
            if len(missing_feats) <= 10:
                print(f"  Missing: {', '.join(missing_feats)}")
            else:
                print(f"  Missing: {len(missing_feats)} features (filled with 0)")

        X = df[features].values.astype(np.float32)
        y = df['_label'].values.astype(int)
        total = len(X)

        print(f"  Data: {total:,} labeled rows | BUY={int((y==1).sum()):,} SELL={int((y==0).sum()):,}")
        print(f"  Class balance: {(y==1).mean():.1%} BUY / {(y==0).mean():.1%} SELL")

        # ── Walk-Forward Evaluation (5 windows) ──
        print(f"\n  +─ Walk-Forward Evaluation (5 windows) ─+")
        wf_results, wf_y_true, wf_y_pred, wf_conf = walk_forward_eval(
            model, model_type, X, y, features, symbol, timeframe, n_windows=5
        )

        if wf_results:
            print(f"  | {'Win':>3} {'N':>6} {'Acc':>7} {'Sharpe':>8} {'PF':>7} {'WR':>7} {'MaxDD':>8} |")
            print(f"  | {'─'*3} {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8} |")
            for r in wf_results:
                pf_str = f"{r['pf']:.2f}" if np.isfinite(r['pf']) and r['pf'] < 100 else "INF"
                print(f"  | {r['window']:>3} {r['n']:>6} {r['acc']:>6.1%} {r['sharpe']:>8.2f} "
                      f"{pf_str:>7} {r['win_rate']:>6.1f}% {r['max_dd']:>7.1f}% |")
            print(f"  +{'─'*50}+")

        # ── Hold-Out Test (last 20%) ──
        split = int(total * 0.80)
        X_test, y_test = X[split:], y[split:]
        print(f"\n  Hold-out test set (last 20%): {len(X_test):,} rows")

        result = evaluate_model(model, model_type, X_test, y_test, features, symbol, timeframe)
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        result['model_type'] = model_type
        result['n_features'] = len(features)
        result['feature_coverage'] = coverage
        result['total_data'] = total

        grand_y_true.extend(y_test)
        grand_y_pred.extend(
            ((model.predict(
                __import__('xgboost').DMatrix(X_test, feature_names=features)
            ) > 0.5).astype(int)) if model_type == 'xgb' else model.predict(X_test)
        )

        # Print detailed results
        print(f"\n  +─ Classification Metrics ────────────────+")
        print(f"  | Accuracy:   {result['accuracy']:>8.2%}                  |")
        print(f"  | Precision:  {result['precision']:>8.2%}                  |")
        print(f"  | Recall:     {result['recall']:>8.2%}                  |")
        print(f"  | F1 Score:   {result['f1']:>8.2%}                  |")
        print(f"  | AUC-ROC:    {result['auc']:>8.4f}                  |")
        print(f"  +─ Per-Class Accuracy ────────────────────+")
        print(f"  | BUY  acc:   {result['buy_acc']:>8.2%}  (n={result['buy_count']:>5})   |")
        print(f"  | SELL acc:   {result['sell_acc']:>8.2%}  (n={result['sell_count']:>5})   |")
        print(f"  | Pred dist:  BUY={result['pred_buy']:>5} SELL={result['pred_sell']:>5}  |")
        print(f"  +────────────────────────────────────────+")

        print(f"\n  +─ Trading Metrics ────────────────────────+")
        pf_str = f"{result['pf']:.2f}" if np.isfinite(result['pf']) and result['pf'] < 100 else "INF"
        print(f"  | Sharpe Ratio:   {result['sharpe']:>8.2f}                |")
        print(f"  | Profit Factor:  {pf_str:>8}                |")
        print(f"  | Win Rate:       {result['win_rate']:>7.1f}%                |")
        print(f"  | Max Drawdown:   {result['max_dd']:>7.1f}%                |")
        print(f"  | Trades:         {result['n_trades']:>7}                 |")
        print(f"  | Avg Confidence: {result['mean_confidence']:>7.1%}                 |")
        print(f"  +─────────────────────────────────────────+")

        # Confusion matrix
        cm = result['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"              Pred SELL  Pred BUY")
        print(f"  Act SELL    {cm[0][0]:>8}  {cm[0][1]:>8}")
        print(f"  Act BUY     {cm[1][0]:>8}  {cm[1][1]:>8}")

        # Confidence analysis
        if result['confidence_analysis']:
            print(f"\n  Confidence Filtering:")
            print(f"  {'Threshold':>10} {'Trades':>8} {'% of Total':>10} {'Accuracy':>10}")
            for c in result['confidence_analysis']:
                print(f"  {c['threshold']:>9.0%} {c['n_trades']:>8} {c['pct']:>9.1f}% {c['accuracy']:>9.1%}")

        all_results.append(result)

    # ===============================================================
    # GRAND SUMMARY
    # ===============================================================
    if not all_results:
        print("\nNo models could be evaluated.")
        return

    print(f"\n\n{'=' * 80}")
    print(f"  GRAND SUMMARY -- {len(all_results)} Models Evaluated")
    print(f"{'=' * 80}")

    # Sort by Sharpe
    sorted_results = sorted(all_results, key=lambda x: -x['sharpe'])

    print(f"\n  {'Symbol':<8} {'TF':<5} {'Type':<5} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7} {'Sharpe':>8} {'PF':>7} {'WR':>7} {'MaxDD':>7}")
    print(f"  {'─'*8} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

    for r in sorted_results:
        pf_str = f"{r['pf']:.2f}" if np.isfinite(r['pf']) and r['pf'] < 100 else "INF"
        print(f"  {r['symbol']:<8} {r['timeframe']:<5} {r['model_type']:<5} "
              f"{r['accuracy']:>6.1%} {r['precision']:>6.1%} {r['recall']:>6.1%} "
              f"{r['f1']:>6.1%} {r['auc']:>6.3f} {r['sharpe']:>8.2f} "
              f"{pf_str:>7} {r['win_rate']:>6.1f}% {r['max_dd']:>6.1f}%")

    # Aggregate stats
    accs = [r['accuracy'] for r in all_results]
    sharpes = [r['sharpe'] for r in all_results]
    pfs = [r['pf'] for r in all_results if np.isfinite(r['pf']) and r['pf'] < 100]
    wrs = [r['win_rate'] for r in all_results]
    aucs = [r['auc'] for r in all_results if r['auc'] > 0]

    print(f"\n  +─ Aggregate Statistics ──────────────────────────────+")
    print(f"  | Models tested:      {len(all_results):>5}                            |")
    print(f"  |                                                      |")
    print(f"  | Accuracy   -- Mean: {np.mean(accs):.1%}  Min: {min(accs):.1%}  Max: {max(accs):.1%}  |")
    print(f"  | Sharpe     -- Mean: {np.mean(sharpes):>5.2f}  Min: {min(sharpes):>5.2f}  Max: {max(sharpes):>5.2f}  |")
    if pfs:
        print(f"  | Profit Fac -- Mean: {np.mean(pfs):>5.2f}  Min: {min(pfs):>5.2f}  Max: {max(pfs):>5.2f}  |")
    print(f"  | Win Rate   -- Mean: {np.mean(wrs):>5.1f}%  Min: {min(wrs):>5.1f}%  Max: {max(wrs):>5.1f}%  |")
    if aucs:
        print(f"  | AUC-ROC    -- Mean: {np.mean(aucs):>5.3f}  Min: {min(aucs):>5.3f}  Max: {max(aucs):>5.3f}  |")
    print(f"  +──────────────────────────────────────────────────────+")

    # By asset class
    print(f"\n  +─ By Asset Class ──────────────────────────────────+")
    for asset in ['crypto', 'forex', 'index']:
        asset_results = [r for r in all_results if get_asset_class(r['symbol']) == asset]
        if asset_results:
            a_acc = np.mean([r['accuracy'] for r in asset_results])
            a_sh = np.mean([r['sharpe'] for r in asset_results])
            a_wr = np.mean([r['win_rate'] for r in asset_results])
            print(f"  | {asset.upper():<8} ({len(asset_results)} models): "
                  f"Acc={a_acc:.1%}  Sharpe={a_sh:.2f}  WR={a_wr:.1f}%    |")
    print(f"  +────────────────────────────────────────────────────+")

    # By timeframe
    print(f"\n  +─ By Timeframe ────────────────────────────────────+")
    for tf in ['M15', 'H1', 'H4', 'D1']:
        tf_results = [r for r in all_results if r['timeframe'] == tf]
        if tf_results:
            t_acc = np.mean([r['accuracy'] for r in tf_results])
            t_sh = np.mean([r['sharpe'] for r in tf_results])
            t_wr = np.mean([r['win_rate'] for r in tf_results])
            print(f"  | {tf:<5} ({len(tf_results)} models): "
                  f"Acc={t_acc:.1%}  Sharpe={t_sh:.2f}  WR={t_wr:.1f}%         |")
    print(f"  +────────────────────────────────────────────────────+")

    # Best and worst
    best = max(all_results, key=lambda x: x['sharpe'])
    worst = min(all_results, key=lambda x: x['sharpe'])
    most_accurate = max(all_results, key=lambda x: x['accuracy'])

    print(f"\n  BEST Sharpe:    {best['symbol']} {best['timeframe']} -- Sharpe={best['sharpe']:.2f} Acc={best['accuracy']:.1%}")
    print(f"  WORST Sharpe:   {worst['symbol']} {worst['timeframe']} -- Sharpe={worst['sharpe']:.2f} Acc={worst['accuracy']:.1%}")
    print(f"  Most Accurate:  {most_accurate['symbol']} {most_accurate['timeframe']} -- Acc={most_accurate['accuracy']:.1%} Sharpe={most_accurate['sharpe']:.2f}")

    # Models above/below thresholds
    profitable = [r for r in all_results if r['sharpe'] > 0]
    strong = [r for r in all_results if r['accuracy'] > 0.55 and r['sharpe'] > 0.5]
    print(f"\n  Profitable (Sharpe > 0): {len(profitable)}/{len(all_results)}")
    print(f"  Strong (Acc > 55% & Sharpe > 0.5): {len(strong)}/{len(all_results)}")

    # Overall combined accuracy
    if grand_y_true:
        grand_acc = accuracy_score(grand_y_true, grand_y_pred)
        print(f"\n  Combined hold-out accuracy (all models): {grand_acc:.2%} ({len(grand_y_true):,} predictions)")

    print(f"\n{'=' * 80}")
    print(f"  Test completed at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
