"""Visualize data augmentations."""
import matplotlib.pyplot as plt
from src.transforms import get_simclr_transforms, get_train_transforms
from PIL import Image
import numpy as np

def visualize_augmentations(image_path, transform, n_samples=5):
    image = Image.open(image_path).convert('RGB')
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(15, 3))

    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i in range(n_samples):
        augmented = transform(image=np.array(image))['image']
        axes[i+1].imshow(augmented.permute(1, 2, 0))
        axes[i+1].set_title(f'Aug {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig('augmentations.png')
    plt.close()
