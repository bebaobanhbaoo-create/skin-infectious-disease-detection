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

# Enable gradient checkpointing for memory efficiency
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to reduce memory usage."""
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Model does not support gradient checkpointing")

def train_with_amp(model, dataloader, criterion, optimizer, scaler, device):
    """Training loop with automatic mixed precision."""
    model.train()
    total_loss = 0
    
    for batch_idx, (view1, view2) in enumerate(dataloader):
        view1, view2 = view1.to(device), view2.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def save_training_state(model, optimizer, scheduler, epoch, loss, path):
    """Save complete training state for resumption."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")
