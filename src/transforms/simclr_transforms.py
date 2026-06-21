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

# Color jitter strength parameter

def get_eval_transforms(img_size=224):
    """Minimal transforms for evaluation and inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_simclr_transforms_v2(img_size=224, strength=1.0):
    """SimCLR v2 augmentation with adjustable strength."""
    s = strength
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.8*s, contrast=0.8*s, 
            saturation=0.8*s, hue=0.2*s, p=0.8
        ),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, int(img_size*0.1)), p=0.5 if img_size > 64 else 0.0),
        A.Solarize(threshold=128, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class ContrastiveTransform:
    """Wrapper to apply same transform twice for contrastive pairs."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            view1 = self.transform(image=image)['image']
            view2 = self.transform(image=image)['image']
        else:
            img_array = np.array(image)
            view1 = self.transform(image=img_array)['image']
            view2 = self.transform(image=img_array)['image']
        return view1, view2
