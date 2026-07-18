# Data Pipeline

## Data Sources
- Dermnet
- ISIC Archive
- HAM10000
- Custom collected images

## Preprocessing
1. Image deduplication (pHash)
2. Resize to 256x256
3. Quality filtering
4. Class balancing

## Augmentation
- SimCLR augmentations for pretraining
- Medical-specific augmentations for fine-tuning
