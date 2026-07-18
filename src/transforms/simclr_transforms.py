import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_simclr_transforms(img_size=224, s=1.0):
    """SimCLR augmentation pipeline."""
    color_jitter = A.ColorJitter(
        brightness=0.8*s, contrast=0.8*s,
        saturation=0.8*s, hue=0.2*s, p=0.8
    )

    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        color_jitter,
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
