"""Train multi-task classifier."""
import argparse
import torch
from src.models import SimCLR
from src.models.multitask_head import MultiTaskHead
from src.losses import MultiTaskLoss
from src.utils import set_seed, get_logger

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(__name__)

    # Load pretrained backbone
    simclr = SimCLR(backbone_name=args.backbone).to(device)

    # Create multi-task head
    head = MultiTaskHead(
        input_dim=simclr.backbone.num_features,
        num_pathogens=args.num_pathogens,
        num_diseases=args.num_diseases
    ).to(device)

    criterion = MultiTaskLoss()
    logger.info("Starting multi-task training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--num_pathogens', type=int, default=3)
    parser.add_argument('--num_diseases', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)

# Support class weighting for imbalanced data
def compute_class_weights(labels, method='inverse'):
    """Compute class weights for imbalanced datasets."""
    import numpy as np
    from collections import Counter
    
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(counts)
    
    if method == 'inverse':
        weights = {cls: n_samples / (n_classes * count) for cls, count in counts.items()}
    elif method == 'effective':
        beta = 0.999
        weights = {}
        for cls, count in counts.items():
            effective_num = 1.0 - np.power(beta, count)
            weights[cls] = (1.0 - beta) / effective_num
    elif method == 'sqrt_inverse':
        weights = {cls: np.sqrt(n_samples / count) for cls, count in counts.items()}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    
    return weights

def create_weighted_sampler(dataset, labels):
    """Create weighted random sampler for balanced batches."""
    from torch.utils.data import WeightedRandomSampler
    import numpy as np
    
    class_weights = compute_class_weights(labels)
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = np.array(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_loss_weights(pathogen_labels, disease_labels):
    """Compute loss weights for multi-task learning."""
    pathogen_weights = compute_class_weights(pathogen_labels)
    disease_weights = compute_class_weights(disease_labels)
    return pathogen_weights, disease_weights
