import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""

    def __init__(self, temperature=0.5, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)

        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        negatives = sim[~mask].view(2 * batch_size, -1)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp):
        """Update temperature scaling parameter."""
        self.temperature = temp
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature

class TemperatureScheduledNTXent(NTXentLoss):
    """NT-Xent loss with temperature annealing."""
    
    def __init__(self, initial_temp=1.0, final_temp=0.1, batch_size=256):
        super().__init__(initial_temp, batch_size)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.current_epoch = 0
        self.total_epochs = 100
    
    def update_temperature(self, epoch, total_epochs=None):
        """Anneal temperature based on training progress."""
        if total_epochs:
            self.total_epochs = total_epochs
        self.current_epoch = epoch
        
        # Cosine annealing
        progress = epoch / self.total_epochs
        cos_progress = (1 + np.cos(np.pi * progress)) / 2
        self.temperature = self.final_temp + (self.initial_temp - self.final_temp) * cos_progress
        return self.temperature
    
    def get_schedule_info(self):
        """Get temperature scheduling information."""
        return {
            'initial': self.initial_temp,
            'final': self.final_temp,
            'current': self.temperature,
            'epoch': self.current_epoch
        }
