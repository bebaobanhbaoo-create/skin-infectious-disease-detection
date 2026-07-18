import pytest
import torch
from src.losses import NTXentLoss, FocalLoss

def test_nt_xent_loss():
    loss_fn = NTXentLoss(temperature=0.5)
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0

def test_focal_loss():
    loss_fn = FocalLoss()
    inputs = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(inputs, targets)
    assert loss.item() >= 0
