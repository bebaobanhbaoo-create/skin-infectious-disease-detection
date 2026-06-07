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

    def get_num_tasks(self):
        """Return number of prediction tasks."""
        return 2
    
    def get_task_names(self):
        """Return names of prediction tasks."""
        return ['pathogen', 'disease']
    
    def forward_pathogen_only(self, x):
        """Get only pathogen predictions."""
        return self.pathogen_head(x)
    
    def forward_disease_only(self, x):
        """Get only disease predictions."""
        return self.disease_head(x)
    
    def get_task_weights(self):
        """Get current task loss weights."""
        return {'pathogen': 1.0, 'disease': 1.0}

class DynamicMultiTaskHead(MultiTaskHead):
    """Multi-task head with learnable task weights."""
    
    def __init__(self, input_dim, num_pathogens, num_diseases):
        super().__init__(input_dim, num_pathogens, num_diseases)
        self.log_vars = nn.Parameter(torch.zeros(2))
    
    def forward_with_uncertainty(self, x):
        pathogen_logits = self.pathogen_head(x)
        disease_logits = self.disease_head(x)
        return pathogen_logits, disease_logits, self.log_vars
    
    def get_learned_weights(self):
        """Get uncertainty-based task weights."""
        weights = torch.exp(-self.log_vars)
        return {'pathogen': weights[0].item(), 'disease': weights[1].item()}
