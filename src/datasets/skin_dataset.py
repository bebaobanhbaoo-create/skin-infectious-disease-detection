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
