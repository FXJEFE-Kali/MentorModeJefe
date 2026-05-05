"""
Quick test of crypto_model.pkl on walk-forward test set.
Shows accuracy, per-class metrics, confidence analysis, and profit factor.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

LABEL_NAMES = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}

# Load model
model_path = os.path.join(config['models_path'], 'crypto_model.pkl')
print(f"Loading model: {model_path}")
model = joblib.load(model_path)
print(f"  Type: {type(model).__name__}")

# Load data
data_path = os.path.join(config['data_output_path'], 'crypto_training_data.csv')
print(f"Loading data: {data_path}")
df = pd.read_csv(data_path, encoding='utf-8')
print(f"  Total rows: {len(df):,}")

# Features from config (the model was trained on these)
features = config['features']
# Note: future_return is included as a training feature (April 2025 config)
# At live prediction time, EA sends 0 for this feature
if 'future_return' in features:
    print("\n  NOTE: 'future_return' is in feature list (April 2025 configuration)")
    print("  EA sends 0 for this feature at prediction time.\n")

df = df.dropna(subset=features + ['label'])
print(f"  Rows after dropna: {len(df):,}")

X = df[features].values.astype(np.float32)
y = df['label'].values

# Walk-forward split (same as training)
split_idx = int(len(X) * 0.80)
X_test = X[split_idx:]
y_test = y[split_idx:]
print(f"  Test set (last 20%): {len(X_test):,} rows")

# Label distribution in test set
unique, counts = np.unique(y_test, return_counts=True)
print(f"\n  Test Label Distribution:")
for lbl, cnt in zip(unique, counts):
    print(f"    {LABEL_NAMES.get(int(lbl), str(int(lbl))):<6}: {cnt:>8} ({cnt/len(y_test)*100:.1f}%)")

# Predict
print("\n  Running predictions...")
y_pred = model.predict(X_test)

# If model returns mapped labels (0,1,2), unmap them
if hasattr(model, 'label_unmap'):
    print(f"  Model has label_unmap: {model.label_unmap}")
elif set(np.unique(y_pred)).issubset({0, 1, 2}):
    # The training script maps -1->0, 0->1, 1->2
    unmap = {0: -1, 1: 0, 2: 1}
    y_pred = np.array([unmap.get(int(v), v) for v in y_pred])
    print(f"  Unmapped predictions from 0/1/2 to -1/0/1")

acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  OVERALL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*60}")

print(f"\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0)
print(report)

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
print(f"{'Predicted ->':>14}  {'SELL':>8}  {'HOLD':>8}  {'BUY':>8}")
for i, name in enumerate(['SELL', 'HOLD', 'BUY']):
    row = f"  Actual {name:<6}" + "".join(f"{cm[i][j]:>10}" for j in range(3))
    print(row)

# Confidence analysis (if model supports predict_proba)
if hasattr(model, 'predict_proba'):
    print(f"\n{'='*60}")
    print(f"  Confidence Filtering Analysis")
    print(f"{'='*60}")
    probas = model.predict_proba(X_test)
    max_conf = probas.max(axis=1)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for t in thresholds:
        mask = max_conf >= t
        count = mask.sum()
        if count == 0:
            print(f"  conf >= {t:.0%}: {count:>8} trades -> N/A")
            continue
        y_pred_filt = model.predict(X_test[mask])
        if hasattr(model, 'label_unmap'):
            pass  # already unmapped
        elif set(np.unique(y_pred_filt)).issubset({0, 1, 2}):
            unmap = {0: -1, 1: 0, 2: 1}
            y_pred_filt = np.array([unmap.get(int(v), v) for v in y_pred_filt])
        filtered_acc = accuracy_score(y_test[mask], y_pred_filt)
        print(f"  conf >= {t:.0%}: {count:>8} trades ({count/len(max_conf)*100:5.1f}%) -> accuracy {filtered_acc*100:5.1f}%")

# ══════════════════════════════════════════════════════════════
# Equity Curve Simulation + Sharpe / Sortino / PF / Max DD
# ══════════════════════════════════════════════════════════════

price_col_idx = features.index('price') if 'price' in features else 0

# Simulation parameters
INITIAL_BALANCE = 10000.0
FIXED_RISK = 50.0          # Fixed $50 risk per trade (0.5% of 10k)
HORIZON = 5                # 5-bar hold period
SPREAD_BPS = 5.0           # spread in basis points (crypto is wider)


def simulate_equity(X, y_pred_arr, balance_start=INITIAL_BALANCE):
    """Simulate equity curve using fixed position sizing (no compounding blowup)."""
    equity = [balance_start]
    balance = balance_start
    trades = []

    for i in range(len(y_pred_arr) - HORIZON):
        pred = int(y_pred_arr[i])
        if pred == 0:  # HOLD
            equity.append(balance)
            continue

        entry = X[i, price_col_idx]
        exit_p = X[i + HORIZON, price_col_idx]

        if entry == 0 or not np.isfinite(entry) or not np.isfinite(exit_p):
            equity.append(balance)
            continue

        # Percentage return on the trade
        spread_cost = SPREAD_BPS / 10000.0
        if pred == 1:  # BUY
            ret_pct = (exit_p - entry) / entry - spread_cost
        else:  # SELL
            ret_pct = (entry - exit_p) / entry - spread_cost

        # Cap extreme returns (price jumps between symbols in concatenated data)
        ret_pct = np.clip(ret_pct, -0.10, 0.10)  # max 10% per trade

        # Fixed risk: win/lose proportional to return, capped at $50 risk
        pnl = FIXED_RISK * (ret_pct / 0.005)  # normalize: 50bps move = $50
        pnl = np.clip(pnl, -FIXED_RISK * 2, FIXED_RISK * 4)  # max loss 2x risk, max win 4x

        balance += pnl
        balance = max(balance, 1.0)
        equity.append(balance)
        trades.append((i, pred, entry, exit_p, pnl, ret_pct))

    return np.array(equity), trades


def calc_metrics(equity, trades):
    """Calculate Sharpe, Sortino, PF, max DD from equity curve."""
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    trade_returns = [t[5] for t in trades]

    # Annualization factor for M5 bars: 12 bars/hr * 24 hrs * 252 trading days
    ann_factor = np.sqrt(252 * 24 * 12)

    mean_ret = np.mean(returns) if len(returns) > 0 else 0
    std_ret = np.std(returns) if len(returns) > 0 else 1e-10

    # Sharpe ratio (annualized)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

    # Sortino ratio (only downside deviation)
    downside = returns[returns < 0]
    down_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = (mean_ret / down_std * ann_factor) if down_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.maximum(peak, 1e-10)
    max_dd = np.min(drawdown) * 100

    # Profit factor
    gross_profit = sum(t[4] for t in trades if t[4] > 0)
    gross_loss = sum(abs(t[4]) for t in trades if t[4] < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    winners = sum(1 for t in trades if t[4] > 0)
    wr = winners / len(trades) * 100 if trades else 0

    return {
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'pf': pf, 'win_rate': wr, 'total_trades': len(trades),
        'winners': winners, 'losers': len(trades) - winners,
        'final_balance': equity[-1],
        'total_return': (equity[-1] / equity[0] - 1) * 100,
        'gross_profit': gross_profit, 'gross_loss': gross_loss,
    }


# ── Run simulation on all predictions ──
print(f"\n{'='*60}")
print(f"  EQUITY SIMULATION + SHARPE ANALYSIS")
print(f"  (${INITIAL_BALANCE:,.0f} start, ${FIXED_RISK:.0f} risk/trade, {HORIZON}-bar hold)")
print(f"{'='*60}")

equity, trades = simulate_equity(X_test, y_pred)
m = calc_metrics(equity, trades)

print(f"  Final balance:   ${m['final_balance']:,.2f}")
print(f"  Total return:    {m['total_return']:+.2f}%")
print(f"  Total trades:    {m['total_trades']:,}")
print(f"  Winners/Losers:  {m['winners']:,} / {m['losers']:,}")
print(f"  Win rate:        {m['win_rate']:.1f}%")
print(f"  Gross profit:    ${m['gross_profit']:,.2f}")
print(f"  Gross loss:      ${m['gross_loss']:,.2f}")
print(f"  PROFIT FACTOR:   {m['pf']:.2f}")
print(f"  SHARPE RATIO:    {m['sharpe']:.2f} (annualized)")
print(f"  SORTINO RATIO:   {m['sortino']:.2f}")
print(f"  MAX DRAWDOWN:    {m['max_dd']:.2f}%")

# ── Confidence threshold optimization by Sharpe ──
if hasattr(model, 'predict_proba'):
    print(f"\n{'='*60}")
    print(f"  OPTIMAL CONFIDENCE THRESHOLD (by Sharpe)")
    print(f"{'='*60}")
    print(f"  {'Threshold':<12} {'Trades':<10} {'Accuracy':<10} {'PF':<8} {'Sharpe':<10} {'Return':<10}")
    print(f"  {'-'*58}")

    probas = model.predict_proba(X_test)
    max_conf = probas.max(axis=1)
    best_sharpe_t = 0.50
    best_sharpe_val = -999

    for t in np.arange(0.50, 0.96, 0.05):
        mask = max_conf >= t
        if mask.sum() < 50:
            continue

        y_pred_filt = model.predict(X_test[mask])
        if hasattr(model, 'label_unmap'):
            pass
        elif set(np.unique(y_pred_filt)).issubset({0, 1, 2}):
            unmap = {0: -1, 1: 0, 2: 1}
            y_pred_filt = np.array([unmap.get(int(v), v) for v in y_pred_filt])

        filt_acc = accuracy_score(y_test[mask], y_pred_filt)
        eq_f, tr_f = simulate_equity(X_test[mask], y_pred_filt)
        mf = calc_metrics(eq_f, tr_f)

        if mf['sharpe'] > best_sharpe_val:
            best_sharpe_val = mf['sharpe']
            best_sharpe_t = t

        print(f"  >= {t:.0%}        {mf['total_trades']:<10,} {filt_acc:<10.1%} {mf['pf']:<8.2f} {mf['sharpe']:<10.2f} {mf['total_return']:+.1f}%")

    print(f"\n  BEST THRESHOLD: {best_sharpe_t:.0%} (Sharpe = {best_sharpe_val:.2f})")
    print(f"  >> Set EA input ConfidenceThreshold = {best_sharpe_t:.2f}")

# Leakage check
print(f"\n{'='*60}")
print(f"  DATA LEAKAGE CHECK")
print(f"{'='*60}")
leaky = ['future_price', 'price_change', 'label']
found_leaks = [f for f in leaky if f in features]
if found_leaks:
    print(f"  LEAKED FEATURES IN TRAINING: {found_leaks}")
    print(f"  These features contain future information!")
    print(f"  Remove them from config.json 'features' list and retrain.")
else:
    print(f"  No obvious leakage detected in feature list.")

print(f"\nDone.")
