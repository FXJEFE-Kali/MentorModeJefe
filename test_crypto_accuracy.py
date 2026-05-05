"""
Quick accuracy test for the saved crypto ensemble model.
Loads crypto_model.pkl and evaluates on the walk-forward test split (last 20%).
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

LABEL_NAMES = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


def main():
    data_path  = os.path.join(config['data_output_path'], 'crypto_training_data.csv')
    model_path = os.path.join(config['models_path'], 'crypto_model.pkl')

    # Load model
    print(f"Loading model from {model_path} ...")
    wrapper = joblib.load(model_path)
    print(f"  Model type: {type(wrapper).__name__}")
    if hasattr(wrapper, 'feature_names'):
        print(f"  Pruned features ({len(wrapper.feature_names)}): {wrapper.feature_names}")
    if hasattr(wrapper, 'all_feature_names'):
        print(f"  Full features: {len(wrapper.all_feature_names)}")

    # Determine which features the model expects
    if hasattr(wrapper, 'all_feature_names'):
        features = wrapper.all_feature_names
    elif hasattr(wrapper, 'feature_names_in_'):
        features = list(wrapper.feature_names_in_)
    else:
        features = config['features']
    print(f"  Using {len(features)} features for prediction")

    # Load data
    print(f"\nLoading data from {data_path} ...")
    data = pd.read_csv(data_path, encoding='utf-8')
    print(f"  Total rows: {len(data)}")

    data = data.dropna(subset=features + ['label'])
    print(f"  Rows after dropna: {len(data)}")

    X = data[features].values
    y = data['label'].values

    # Walk-forward split: test on last 20%
    n = len(X)
    split_idx = int(n * 0.80)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    print(f"  Test set (last 20%): {len(X_test)} rows")

    # Predict using wrapper (handles feature pruning internally)
    print("\nRunning predictions ...")
    y_pred = wrapper.predict(X_test)

    # If model outputs 0-indexed labels (0,1,2), unmap to (-1,0,1)
    label_unmap = {0: -1, 1: 0, 2: 1}
    if hasattr(wrapper, 'label_unmap'):
        label_unmap = wrapper.label_unmap
    if set(np.unique(y_pred)).issubset({0, 1, 2}):
        print("  Unmapping model labels {0,1,2} -> {-1,0,1}")
        y_pred = np.array([label_unmap[int(v)] for v in y_pred])

    # Overall accuracy
    acc = (y_pred == y_test).mean()
    print(f"\n{'='*50}")
    print(f"  OVERALL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")

    # Per-class report
    labels = sorted(LABEL_NAMES.keys())
    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=[LABEL_NAMES[l] for l in labels]
    )
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    names = [LABEL_NAMES[l] for l in labels]
    header = f"{'Predicted ->':>12}" + "".join(f"{n:>8}" for n in names)
    print(f"\n{'='*50}")
    print(f"  Confusion Matrix (Walk-Forward Test)")
    print(f"{'='*50}")
    print(header)
    for i, row_label in enumerate(names):
        row = f"{'Actual '+row_label:>12}" + "".join(f"{cm[i][j]:>8}" for j in range(len(names)))
        print(row)

    # Per-symbol accuracy (if symbol column exists)
    all_features_plus = features + ['label']
    if 'symbol' in data.columns:
        print(f"\n{'='*50}")
        print(f"  Per-Symbol Accuracy")
        print(f"{'='*50}")
        symbols_test = data['symbol'].values[split_idx:]
        for sym in sorted(set(symbols_test)):
            mask = symbols_test == sym
            sym_acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"  {sym:<10}: {sym_acc:.4f} ({sym_acc*100:.2f}%)  [{mask.sum()} rows]")

    # Confidence analysis (if model supports predict_proba)
    try:
        print(f"\n{'='*50}")
        print(f"  Confidence Analysis")
        print(f"{'='*50}")
        probas = wrapper.predict_proba(X_test)
        max_conf = probas.max(axis=1)

        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        for t in thresholds:
            mask = max_conf >= t
            count = mask.sum()
            pct = count / len(max_conf) * 100
            if count > 0:
                filtered_acc = (y_pred[mask] == y_test[mask]).mean() * 100
            else:
                filtered_acc = 0
            print(f"  conf >= {t:.0%}: {count:>7} trades ({pct:5.1f}%) -> accuracy {filtered_acc:5.1f}%")
    except Exception as e:
        print(f"  Confidence analysis skipped: {e}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
