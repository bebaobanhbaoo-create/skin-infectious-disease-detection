from torch.utils.data import Dataset
from PIL import Image
import os

class BaseImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_num_classes(self):
        """Return number of unique classes in dataset."""
        return len(set(self.get_labels()))
    
    def get_class_distribution(self):
        """Get count of samples per class."""
        from collections import Counter
        return dict(Counter(self.get_labels()))
    
    def get_sample_weights(self):
        """Compute sample weights for balanced sampling."""
        import numpy as np
        labels = self.get_labels()
        class_counts = self.get_class_distribution()
        n_samples = len(labels)
        n_classes = len(class_counts)
        
        weights = []
        for label in labels:
            weight = n_samples / (n_classes * class_counts[label])
            weights.append(weight)
        return np.array(weights)
    
    def split_by_class(self, train_ratio=0.8):
        """Stratified split maintaining class distribution."""
        from collections import defaultdict
        import random
        
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            class_indices[label].append(idx)
        
        train_indices, val_indices = [], []
        for label, indices in class_indices.items():
            random.shuffle(indices)
            split_point = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_point])
            val_indices.extend(indices[split_point:])
        
        return train_indices, val_indices
