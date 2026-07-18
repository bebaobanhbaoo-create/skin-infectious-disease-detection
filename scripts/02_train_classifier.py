"""Fine-tune classifier on pretrained features."""
import argparse
import torch
from src.models import SimCLR, LinearClassifier
from src.utils import set_seed, get_logger, load_checkpoint

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained SimCLR
    simclr = SimCLR(backbone_name=args.backbone).to(device)
    load_checkpoint(args.pretrained_path, simclr)
    simclr.eval()

    # Freeze backbone
    for param in simclr.parameters():
        param.requires_grad = False

    # Train classifier
    classifier = LinearClassifier(
        input_dim=simclr.backbone.num_features,
        num_classes=args.num_classes
    ).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
