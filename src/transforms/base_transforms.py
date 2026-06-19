import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
A.RandomBrightnessContrast(p=0.5),

def get_test_transforms(img_size=224):
    return get_val_transforms(img_size)

def get_strong_augmentations(img_size=224):
    """Strong augmentation pipeline for self-supervised learning."""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.08, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=7),
            A.MotionBlur(blur_limit=7),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(color_shift=(0.01, 0.05)),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(img_size=224, n_augments=5):
    """Test-time augmentation transforms."""
    base_transform = get_val_transforms(img_size)
    
    tta_transforms = [base_transform]
    tta_transforms.append(A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    tta_transforms.append(A.Compose([
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    return tta_transforms[:n_augments]
