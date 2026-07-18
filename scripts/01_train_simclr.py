"""Train SimCLR for self-supervised pretraining."""
import argparse
import torch
from torch.utils.data import DataLoader
from src.models import SimCLR
from src.losses import NTXentLoss
from src.datasets import SimCLRDataset
from src.transforms import get_simclr_transforms
from src.utils import set_seed, get_logger

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for view1, view2 in dataloader:
        view1, view2 = view1.to(device), view2.to(device)

        z1 = model(view1)
        z2 = model(view2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main(args):
    set_seed(args.seed)
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimCLR(backbone_name=args.backbone).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f"Training SimCLR with {args.backbone} backbone")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
import wandb
    logger.info(f'Device: {device}')
