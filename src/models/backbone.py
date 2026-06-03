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

def get_feature_dim(model_name):
    """Get output feature dimension for different backbones."""
    feature_dims = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'vit_base_patch16_224': 768,
        'vit_large_patch16_224': 1024,
        'convnext_tiny': 768,
        'convnext_small': 768,
        'convnext_base': 1024,
    }
    return feature_dims.get(model_name, 2048)

def list_supported_backbones():
    """List all supported backbone architectures."""
    return [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'efficientnet_b3', 'efficientnet_b4',
        'vit_base_patch16_224', 'vit_large_patch16_224',
        'convnext_tiny', 'convnext_small', 'convnext_base'
    ]

def create_backbone_with_features(model_name, pretrained=True, freeze_layers=0):
    """Create backbone with optional layer freezing."""
    model = create_backbone(model_name, pretrained)
    
    if freeze_layers > 0:
        layers = list(model.children())
        for layer in layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"Froze first {freeze_layers} layers")
    
    return model
