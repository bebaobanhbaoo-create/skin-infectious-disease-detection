import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def get_gamma(self):
        return self.gamma

    def set_alpha(self, alpha):
        """Update class balancing factor."""
        self.alpha = alpha
    
    def set_gamma(self, gamma):
        """Update focusing parameter."""
        self.gamma = gamma
    
    def get_params(self):
        """Get current loss parameters."""
        return {'alpha': self.alpha, 'gamma': self.gamma}

class AdaptiveFocalLoss(FocalLoss):
    """Focal loss with adaptive gamma based on training progress."""
    
    def __init__(self, alpha=1, initial_gamma=2, final_gamma=5, reduction='mean'):
        super().__init__(alpha, initial_gamma, reduction)
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.current_epoch = 0
        self.total_epochs = 100
    
    def update_gamma(self, epoch, total_epochs=None):
        """Update gamma based on training progress."""
        if total_epochs:
            self.total_epochs = total_epochs
        self.current_epoch = epoch
        
        progress = epoch / self.total_epochs
        self.gamma = self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress
        return self.gamma
    
    def forward(self, inputs, targets):
        """Forward with current gamma value."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
