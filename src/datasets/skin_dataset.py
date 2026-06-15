from .base_dataset import BaseImageDataset
import os
from pathlib import Path

class SkinDiseaseDataset(BaseImageDataset):
    def __init__(self, data_dir, split='train', transform=None):
        super().__init__(data_dir, transform)
        self.split = split
        self._load_samples()

    def _load_samples(self):
        split_dir = Path(self.data_dir) / self.split
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label))
        for img_path in class_dir.glob('*.png'):
            self.samples.append((str(img_path), label))

    def get_class_weights(self):
        pass

    def get_class_names(self):
        """Return sorted list of unique class names."""
        return sorted(set([s[1] for s in self.samples]))
    
    def get_class_to_idx(self):
        """Return mapping from class name to index."""
        return {name: idx for idx, name in enumerate(self.get_class_names())}
    
    def get_idx_to_class(self):
        """Return mapping from index to class name."""
        return {idx: name for name, idx in self.get_class_to_idx().items()}
    
    def filter_by_classes(self, class_names):
        """Return subset containing only specified classes."""
        filtered_samples = [(path, label) for path, label in self.samples 
                           if label in class_names]
        new_dataset = SkinDiseaseDataset.__new__(SkinDiseaseDataset)
        new_dataset.data_dir = self.data_dir
        new_dataset.split = self.split
        new_dataset.transform = self.transform
        new_dataset.samples = filtered_samples
        return new_dataset
    
    def get_pathogen_labels(self):
        """Extract pathogen type from disease labels."""
        pathogen_map = {
            'herpes': 'virus', 'warts': 'virus', 'chickenpox': 'virus',
            'impetigo': 'bacteria', 'cellulitis': 'bacteria', 
            'candidiasis': 'fungus', 'tinea': 'fungus', 'ringworm': 'fungus'
        }
        return [pathogen_map.get(label.lower(), 'unknown') for _, label in self.samples]
