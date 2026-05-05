"""
Train ensemble model for crypto pairs (BTCUSD, ETHUSD, XRPUSD)
- RandomForest + GradientBoosting + XGBoost soft-voting ensemble
- Walk-forward validation (train on older data, test on newer)
- Time-decay sample weights (recent data weighted higher)
- Auto feature pruning (drop features with <1% importance)
- Confusion matrix + classification report + confidence analysis
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

os.makedirs(config['log_path'],   exist_ok=True)
os.makedirs(config['models_path'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config['log_path'], 'train_crypto_model.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

LABEL_NAMES = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


def print_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [LABEL_NAMES.get(l, str(l)) for l in labels]
    header = f"{'Predicted ->':>12}" + "".join(f"{n:>8}" for n in names)
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(header)
    for i, row_label in enumerate(names):
        row = f"{'Actual '+row_label:>12}" + "".join(f"{cm[i][j]:>8}" for j in range(len(names)))
        print(row)
        logging.info(row)
    if save_path:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info(f"Confusion matrix saved -> {save_path}")


def print_feature_importance(model, features, top_n=20):
    if hasattr(model, 'estimators_'):
        importances = np.zeros(len(features))
        count = 0
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                importances += est.feature_importances_
                count += 1
        if count > 0:
            importances /= count
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    indices = np.argsort(importances)[::-1]
    print(f"\n  Top {min(top_n, len(features))} Feature Importances (ensemble avg)")
    print(f"  {'-'*45}")
    for rank in range(min(top_n, len(features))):
        idx = indices[rank]
        line = f"  {rank+1:>2}. {features[idx]:<22} {importances[idx]:.4f}"
        logging.info(line)
        print(line)
    return importances


def analyze_confidence(model, X_test, y_test):
    probas = model.predict_proba(X_test)
    max_conf = probas.max(axis=1)
    print(f"\n  Confidence Distribution")
    print(f"  {'-'*55}")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for t in thresholds:
        mask = max_conf >= t
        count = mask.sum()
        pct = count / len(max_conf) * 100
        if count > 0:
            y_pred_filtered = model.predict(X_test[mask])
            acc = (y_pred_filtered == y_test[mask]).mean() * 100
        else:
            acc = 0
        line = f"  conf >= {t:.0%}: {count:>7} trades ({pct:5.1f}%) -> accuracy {acc:5.1f}%"
        logging.info(line)
        print(line)
    return max_conf


def prune_features(importances, features, threshold=0.01):
    """Return indices of features with importance >= threshold."""
    keep = [i for i, imp in enumerate(importances) if imp >= threshold]
    dropped = [features[i] for i in range(len(features)) if i not in keep]
    if dropped:
        logging.info(f"Pruning {len(dropped)} low-importance features: {dropped}")
        print(f"\n  Pruned {len(dropped)} features (<{threshold:.0%} importance): {dropped}")
    return keep


def main():
    data_path  = os.path.join(config['data_output_path'], 'crypto_training_data.csv')
    model_path = os.path.join(config['models_path'],      'crypto_model.pkl')
    cm_path    = os.path.join(config['models_path'],      'crypto_confusion_matrix.png')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"crypto_training_data.csv not found at {data_path}")

    data = pd.read_csv(data_path, encoding='utf-8')
    logging.info(f"Loaded {len(data)} training rows.")

    features = config['features']
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if 'label' not in data.columns:
        raise ValueError("Missing label column")

    # Extract sample weights if available
    has_weights = 'sample_weight' in data.columns
    if has_weights:
        logging.info("Using time-decay sample weights")

    data = data.dropna(subset=features + ['label'])
    logging.info(f"Rows after dropna: {len(data)}")

    X = data[features].values
    y = data['label'].values
    weights = data['sample_weight'].values if has_weights else None

    # XGBoost needs 0-indexed labels
    label_map = {-1: 0, 0: 1, 1: 2}
    label_unmap = {0: -1, 1: 0, 2: 1}
    y_mapped = np.array([label_map[int(v)] for v in y])

    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Label Distribution ({len(features)} features)")
    print(f"  {'-'*30}")
    for lbl, cnt in zip(unique, counts):
        print(f"  {LABEL_NAMES.get(int(lbl), str(lbl)):<6}: {cnt:>8} ({cnt/len(y)*100:.1f}%)")

    if has_weights:
        wt_unique, wt_counts = np.unique(weights, return_counts=True)
        print(f"\n  Sample Weight Distribution")
        print(f"  {'-'*30}")
        for w, c in zip(wt_unique, wt_counts):
            print(f"  weight={w:.2f}: {c:>8} ({c/len(weights)*100:.1f}%)")

    # ── WALK-FORWARD SPLIT ──────────────────────────────────────
    # Train on first 80% (older data), test on last 20% (newer data)
    # This prevents future data leakage unlike random split
    n = len(X)
    split_idx = int(n * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]
    w_train = weights[:split_idx] if has_weights else None
    w_test = weights[split_idx:] if has_weights else None
    logging.info(f"Walk-forward split: Train={len(X_train)} (older), Test={len(X_test)} (newer)")
    print(f"\n  Walk-forward: Train {len(X_train)} oldest -> Test {len(X_test)} newest")

    # ── PHASE 1: Train RF to get feature importance for pruning ─
    print(f"\n  Phase 1: Quick RF for feature importance...")
    rf_scout = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
        class_weight='balanced',
    )
    rf_scout.fit(X_train, y_train, sample_weight=w_train)
    importances = rf_scout.feature_importances_

    # Print all feature importances
    print_feature_importance(rf_scout, features, top_n=len(features))

    # Auto-prune features with <1% importance
    keep_idx = prune_features(importances, features, threshold=0.01)
    pruned_features = [features[i] for i in keep_idx]
    logging.info(f"Keeping {len(pruned_features)}/{len(features)} features after pruning")
    print(f"  Keeping {len(pruned_features)}/{len(features)} features")

    X_train_p = X_train[:, keep_idx]
    X_test_p = X_test[:, keep_idx]

    # ── PHASE 2: Train individual models on pruned features ─────
    print(f"\n  Phase 2: Training 3 models on {len(pruned_features)} features...")

    # RandomForest
    print("  [1/3] RandomForest (500 trees)...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=30, min_samples_split=5,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
        class_weight='balanced',
    )
    rf.fit(X_train_p, y_train, sample_weight=w_train)
    rf_acc = (rf.predict(X_test_p) == y_test).mean()
    print(f"        RF accuracy: {rf_acc:.4f}")
    logging.info(f"RF accuracy: {rf_acc:.4f}")

    # GradientBoosting
    print("  [2/3] GradientBoosting (300 trees)...")
    gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        min_samples_split=10, min_samples_leaf=5,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_train_p, y_train, sample_weight=w_train)
    gbm_acc = (gbm.predict(X_test_p) == y_test).mean()
    print(f"        GBM accuracy: {gbm_acc:.4f}")
    logging.info(f"GBM accuracy: {gbm_acc:.4f}")

    # XGBoost
    print("  [3/3] XGBoost (500 trees)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1,
    )
    xgb_clf.fit(X_train_p, y_train, sample_weight=w_train)
    xgb_acc = (xgb_clf.predict(X_test_p) == y_test).mean()
    print(f"        XGB accuracy: {xgb_acc:.4f}")
    logging.info(f"XGB accuracy: {xgb_acc:.4f}")

    # ── PHASE 3: Ensemble (reuse pre-trained models to save memory) ──
    print("  Building ensemble (soft vote, reusing trained models)...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('xgb', xgb_clf)],
        voting='soft',
        n_jobs=1,
    )
    # Manually set fitted state to avoid re-training
    ensemble.estimators_ = [rf, gbm, xgb_clf]
    ensemble.named_estimators_ = {'rf': rf, 'gbm': gbm, 'xgb': xgb_clf}
    ensemble.le_ = None
    ensemble.classes_ = rf.classes_

    y_pred_mapped = ensemble.predict(X_test_p)
    ens_acc = (y_pred_mapped == y_test).mean()
    print(f"        Ensemble accuracy: {ens_acc:.4f}")
    logging.info(f"Ensemble accuracy: {ens_acc:.4f}")

    # ── Results comparison ─────────────────────────────────────
    print(f"\n  {'='*40}")
    print(f"  Model Comparison (walk-forward)")
    print(f"  {'='*40}")
    print(f"  RandomForest:      {rf_acc:.4f}")
    print(f"  GradientBoosting:  {gbm_acc:.4f}")
    print(f"  XGBoost:           {xgb_acc:.4f}")
    print(f"  Ensemble (vote):   {ens_acc:.4f}")

    # Unmap labels for display
    y_test_orig = np.array([label_unmap[v] for v in y_test])
    y_pred_orig = np.array([label_unmap[v] for v in y_pred_mapped])

    labels_orig = sorted(LABEL_NAMES.keys())
    report = classification_report(
        y_test_orig, y_pred_orig,
        target_names=[LABEL_NAMES[l] for l in labels_orig]
    )
    logging.info(f"Classification Report:\n{report}")
    print(f"\n{report}")

    print_confusion_matrix(y_test_orig, y_pred_orig, labels_orig,
                          "Crypto Ensemble -- Walk-Forward Test", cm_path)

    # Feature importance on pruned features
    print_feature_importance(ensemble, pruned_features)

    # Confidence analysis
    analyze_confidence(ensemble, X_test_p, y_test)

    # ── Save model with metadata ───────────────────────────────
    class CryptoEnsembleWrapper:
        def __init__(self, model, label_unmap, feature_indices, feature_names, all_feature_names):
            self.model = model
            self.label_unmap = label_unmap
            self.feature_indices = feature_indices      # indices into full feature list
            self.feature_names = feature_names           # pruned feature names
            self.all_feature_names = all_feature_names   # full config feature names
            self.classes_ = np.array(sorted(label_unmap.values()))
            self.n_features_in_ = len(all_feature_names)  # server sends full features

        def predict(self, X):
            X_arr = np.array(X)
            if X_arr.shape[-1] == len(self.all_feature_names):
                X_arr = X_arr[:, self.feature_indices]
            mapped = self.model.predict(X_arr)
            return np.array([self.label_unmap[v] for v in mapped])

        def predict_proba(self, X):
            X_arr = np.array(X)
            if X_arr.shape[-1] == len(self.all_feature_names):
                X_arr = X_arr[:, self.feature_indices]
            return self.model.predict_proba(X_arr)

    wrapper = CryptoEnsembleWrapper(
        ensemble, label_unmap, keep_idx, pruned_features, features
    )
    joblib.dump(wrapper, model_path)
    logging.info(f"Ensemble saved -> {model_path}")
    print(f"\n  Model saved -> {model_path}")
    print(f"  Full features: {len(features)}, Pruned to: {len(pruned_features)}")
    print(f"  Pruned features used: {pruned_features}")


if __name__ == '__main__':
    main()
