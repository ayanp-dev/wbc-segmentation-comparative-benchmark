import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate multiclass evaluation metrics including overall accuracy, macro F1,
    per-class sensitivity, specificity, and confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    overall_acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    num_classes = len(class_names)
    total_samples = np.sum(cm)

    for i, c_name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / total_samples if total_samples > 0 else 0.0

        per_class_metrics[c_name] = {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': acc
        }

    return {
        'overall_accuracy': overall_acc,
        'macro_f1': macro_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
