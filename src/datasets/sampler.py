from torch.utils.data import Sampler
import numpy as np

class BalancedSampler(Sampler):
    """Balanced sampling for imbalanced datasets."""

    def __init__(self, labels, num_samples=None):
        self.labels = np.array(labels)
        self.num_samples = num_samples or len(labels)

        label_counts = np.bincount(self.labels)
        weights = 1.0 / label_counts[self.labels]
        self.weights = weights / weights.sum()

    def __iter__(self):
        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
