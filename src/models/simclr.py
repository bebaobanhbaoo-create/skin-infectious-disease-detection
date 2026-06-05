import torch
import torch.nn as nn
from .backbone import create_backbone, get_backbone_output_dim
from .projection_head import ProjectionHead

class SimCLR(nn.Module):
    """SimCLR model for self-supervised learning."""

    def __init__(self, backbone_name='resnet50', projection_dim=128):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained=True)
        backbone_dim = get_backbone_output_dim(backbone_name)
        self.projection_head = ProjectionHead(backbone_dim, output_dim=projection_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return nn.functional.normalize(projections, dim=1)

    def get_features(self, x):
        """Get backbone features without projection."""
        return self.backbone(x)
# Add type hints
        self.device = device

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Enable gradient computation for backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")
    
    def freeze_projection_head(self):
        """Freeze projection head during linear evaluation."""
        for param in self.projection_head.parameters():
            param.requires_grad = False
    
    def get_backbone_features(self, x):
        """Extract features before projection head."""
        return self.backbone(x)
    
    def get_embedding_dim(self):
        """Return the dimension of output embeddings."""
        return self.projection_head.net[-1].out_features
    
    def load_pretrained_backbone(self, checkpoint_path):
        """Load only backbone weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        backbone_state = {k.replace('backbone.', ''): v 
                         for k, v in checkpoint['model_state_dict'].items() 
                         if k.startswith('backbone.')}
        self.backbone.load_state_dict(backbone_state)
        print(f"Loaded backbone from {checkpoint_path}")
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {'trainable': trainable, 'total': total, 'frozen': total - trainable}
