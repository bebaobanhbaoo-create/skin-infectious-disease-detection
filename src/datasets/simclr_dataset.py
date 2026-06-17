from torch.utils.data import Dataset
from PIL import Image

class SimCLRDataset(Dataset):
    """Dataset for SimCLR self-supervised learning."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transform twice for positive pair
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2

    def set_transform(self, transform):
        self.transform = transform

    def __repr__(self):
        """Return informative string representation."""
        return f'SimCLRDataset(size={len(self)}, transform={self.transform is not None})'
    
    def set_transform(self, transform):
        """Update augmentation transform."""
        self.transform = transform
    
    def get_image_paths(self):
        """Return list of all image paths."""
        return self.image_paths.copy()
    
    def create_subset(self, indices):
        """Create a subset dataset with given indices."""
        subset_paths = [self.image_paths[i] for i in indices]
        return SimCLRDataset(subset_paths, self.transform)
    
    def get_statistics(self):
        """Compute dataset statistics for normalization."""
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
        
        means = []
        stds = []
        
        for path in tqdm(self.image_paths[:1000], desc="Computing stats"):
            img = np.array(Image.open(path).convert('RGB')) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        
        return {
            'mean': np.mean(means, axis=0).tolist(),
            'std': np.mean(stds, axis=0).tolist()
        }
