# Experiments

## Scenarios

### Scenario 1: Supervised Baseline
- ImageNet pretrained ResNet-50
- Standard cross-entropy loss
- Baseline for comparison

### Scenario 2: SimCLR + Linear
- SimCLR pretraining
- Frozen backbone + linear classifier
- Evaluate representation quality

### Scenario 3: SimCLR + Multi-task
- SimCLR pretraining
- Fine-tune with multi-task objective
- Best expected performance
