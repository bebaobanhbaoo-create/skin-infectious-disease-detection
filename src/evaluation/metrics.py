"""Evaluation metrics for classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

def compute_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except:
            pass

    return metrics

def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

def get_confusion_matrix(y_true, y_pred, normalize=True):
    """Get confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    return cm

def get_all_metrics():
    """Return list of all available evaluation metrics."""
    return ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']

def compute_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_balanced_accuracy(y_true, y_pred):
    """Calculate balanced accuracy for imbalanced datasets."""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)

def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple folds or runs."""
    import numpy as np
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    return aggregated
