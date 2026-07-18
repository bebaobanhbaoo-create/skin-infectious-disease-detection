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
