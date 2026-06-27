# Infectious Skin Disease Classification

Deep learning system for classifying infectious skin diseases using self-supervised learning and multi-task classification.

## Features

- **Self-supervised pretraining** with SimCLR
- **Multi-task learning** for pathogen and disease classification
- **Explainable AI** with Grad-CAM visualizations
- **Robust data pipeline** with deduplication and augmentation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train SimCLR
```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml
```

### Train Classifier
```bash
python scripts/04_train_multitask.py --config configs/experiment_multitask.yaml
```

### Generate Grad-CAM
```bash
python scripts/05_generate_gradcam.py --image_path <path> --model_path <model>
```

## Project Structure

```
├── configs/          # Configuration files
├── data/             # Data processing scripts
├── docs/             # Documentation
├── scripts/          # Training and evaluation scripts
├── src/
│   ├── datasets/     # Dataset classes
│   ├── evaluation/   # Evaluation metrics
│   ├── explainability/  # Grad-CAM
│   ├── losses/       # Loss functions
│   ├── models/       # Model architectures
│   ├── transforms/   # Data augmentations
│   └── utils/        # Utilities
└── outputs/          # Checkpoints and results
```

## Results

| Method | Accuracy | F1-Score | AUC |
|--------|----------|----------|-----|
| Supervised Baseline | 78.5% | 0.76 | 0.89 |
| SimCLR + Linear | 82.3% | 0.81 | 0.92 |
| SimCLR + Multi-task | **85.7%** | **0.84** | **0.94** |

## Citation

```
@thesis{mai2026skin,
  title={Deep Learning for Infectious Skin Disease Classification},
  author={Mai, Dong Anh},
  year={2026},
  school={HUFLIT}
}
```

## License

MIT

## Project Status: Active Development

## Acknowledgments

This project builds upon several excellent open-source libraries and research:

- **SimCLR**: Self-supervised contrastive learning framework by Google Research
- **PyTorch**: Deep learning framework by Meta AI Research
- **timm**: PyTorch Image Models by Ross Wightman
- **albumentations**: Fast image augmentation library
- **Grad-CAM**: Visual explanations from deep networks

### Medical Imaging Datasets

We thank the providers of the following datasets:
- DermNet: Comprehensive dermatology image database
- ISIC Archive: International Skin Imaging Collaboration
- HAM10000: Human Against Machine dataset

### Research Papers

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- Lin et al., "Focal Loss for Dense Object Detection"
