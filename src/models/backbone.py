import torch.nn as nn
import timm

def create_backbone(model_name='resnet50', pretrained=True):
    """Create backbone encoder from timm."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return model

def get_backbone_output_dim(model_name='resnet50'):
    """Get output dimension of backbone."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    return model.num_features

def list_available_backbones():
    return ['resnet18', 'resnet50', 'efficientnet_b0']
