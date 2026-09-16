import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_metrics(y_true, y_pred):
    """Compute AUC and accuracy for binary prediction."""
    try:
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5
    acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
    return auc, acc
