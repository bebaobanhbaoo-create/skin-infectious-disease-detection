import torch.nn as nn

class MultiTaskHead(nn.Module):
    """Multi-task classification head."""

    def __init__(self, input_dim, num_pathogens, num_diseases):
        super().__init__()
        self.pathogen_head = nn.Linear(input_dim, num_pathogens)
        self.disease_head = nn.Linear(input_dim, num_diseases)

    def forward(self, x):
        pathogen_logits = self.pathogen_head(x)
        disease_logits = self.disease_head(x)
        return pathogen_logits, disease_logits
