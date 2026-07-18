"""Generate Grad-CAM visualizations for model predictions."""
import argparse
import torch
from src.models import SimCLR
from src.explainability import GradCAM
from src.explainability.visualize_cam import visualize_gradcam
from PIL import Image
from src.transforms import get_val_transforms

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimCLR(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.model_path))

    # Get target layer (last conv layer)
    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # Load and preprocess image
    transform = get_val_transforms()
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    cam = gradcam.generate(input_tensor)
    visualize_gradcam(input_tensor, cam, save_path=args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--output_path', default='gradcam_output.png')
    args = parser.parse_args()
    main(args)
