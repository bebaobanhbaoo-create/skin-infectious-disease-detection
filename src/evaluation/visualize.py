"""Visualization for evaluation results."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curves for each class."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def save_all_plots(results, output_dir, format='png', dpi=150):
    """Save all visualization plots to output directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if 'confusion_matrix' in results:
        plot_confusion_matrix(
            results['confusion_matrix'],
            results.get('class_names', None),
            save_path=os.path.join(output_dir, f'confusion_matrix.{format}')
        )
    
    if 'roc_data' in results:
        plot_roc_curves(
            results['roc_data']['y_true'],
            results['roc_data']['y_prob'],
            results.get('class_names', None),
            save_path=os.path.join(output_dir, f'roc_curves.{format}')
        )

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_class_distribution(labels, class_names=None, save_path=None):
    """Visualize class distribution in dataset."""
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    counts = Counter(labels)
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]
    
    if class_names:
        classes = [class_names[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(classes)), values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
