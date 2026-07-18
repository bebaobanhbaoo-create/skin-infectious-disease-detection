import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_gradcam(image, cam, alpha=0.5, save_path=None):
    """Overlay Grad-CAM heatmap on image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if len(image.shape) == 4:
        image = image.squeeze(0)
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)

    image = image.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = np.uint8(255 * image)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM')
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')

    for ax in axes:
        ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
