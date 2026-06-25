import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def get_best_checkpoint(checkpoint_dir):
    pass

def get_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint in directory."""
    import glob
    import os
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def get_best_checkpoint(checkpoint_dir, metric='val_loss', mode='min'):
    """Find best checkpoint based on metric in filename."""
    import glob
    import os
    import re
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        return None
    
    best_ckpt = None
    best_value = float('inf') if mode == 'min' else float('-inf')
    
    for ckpt in checkpoints:
        match = re.search(f'{metric}=([0-9.]+)', ckpt)
        if match:
            value = float(match.group(1))
            if (mode == 'min' and value < best_value) or (mode == 'max' and value > best_value):
                best_value = value
                best_ckpt = ckpt
    
    return best_ckpt

class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""
    
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, state, filename):
        """Save checkpoint and cleanup old ones."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        self._cleanup()
        return path
    
    def _cleanup(self):
        """Remove oldest checkpoints if exceeding limit."""
        import glob
        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, '*.pth')),
            key=os.path.getmtime
        )
        while len(checkpoints) > self.max_checkpoints:
            os.remove(checkpoints.pop(0))
