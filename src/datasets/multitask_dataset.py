from torch.utils.data import Dataset
from PIL import Image

class MultiTaskDataset(Dataset):
    """Dataset with multiple labels per sample."""

    def __init__(self, image_paths, pathogen_labels, disease_labels, transform=None):
        self.image_paths = image_paths
        self.pathogen_labels = pathogen_labels
        self.disease_labels = disease_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.pathogen_labels[idx], self.disease_labels[idx]
