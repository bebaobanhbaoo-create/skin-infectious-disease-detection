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
