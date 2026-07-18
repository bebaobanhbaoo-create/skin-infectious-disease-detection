import pytest
import torch
from src.models import SimCLR
from src.models.projection_head import ProjectionHead

def test_simclr_forward():
    model = SimCLR(backbone_name='resnet18')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 128)

def test_projection_head():
    head = ProjectionHead(512, 2048, 128)
    x = torch.randn(4, 512)
    out = head(x)
    assert out.shape == (4, 128)
