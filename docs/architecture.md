# Model Architecture

## Overview
Two-stage approach for infectious skin disease classification:
1. Self-supervised pretraining with SimCLR
2. Multi-task fine-tuning for pathogen and disease classification

## SimCLR Pretraining
- Backbone: ResNet-50
- Projection head: 2-layer MLP (2048 -> 128)
- Loss: NT-Xent with temperature 0.5
- Augmentations: Random crop, color jitter, Gaussian blur

## Multi-task Classification
- Shared backbone from SimCLR
- Two classification heads:
  - Pathogen type (virus/bacteria/fungus)
  - Specific disease (10 classes)
- Combined loss with configurable weights
