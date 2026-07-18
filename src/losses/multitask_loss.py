import torch.nn as nn

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(self, pathogen_weight=1.0, disease_weight=1.0):
        super().__init__()
        self.pathogen_weight = pathogen_weight
        self.disease_weight = disease_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pathogen_logits, disease_logits, pathogen_labels, disease_labels):
        pathogen_loss = self.ce_loss(pathogen_logits, pathogen_labels)
        disease_loss = self.ce_loss(disease_logits, disease_labels)

        total_loss = (self.pathogen_weight * pathogen_loss +
                      self.disease_weight * disease_loss)

        return total_loss, pathogen_loss, disease_loss
