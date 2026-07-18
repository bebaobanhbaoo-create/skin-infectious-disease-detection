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
