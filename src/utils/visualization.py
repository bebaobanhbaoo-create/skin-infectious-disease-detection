import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
def plot_confusion_matrix(cm, classes, save_path=None):
    pass
