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
