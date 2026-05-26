import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear')
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()

    def batch_generate(self, input_batch, target_classes=None):
        """Generate Grad-CAM for a batch of images."""
        batch_size = input_batch.shape[0]
        cams = []
        
        for i in range(batch_size):
            single_input = input_batch[i:i+1]
            target = target_classes[i] if target_classes is not None else None
            cam = self.generate(single_input, target)
            cams.append(cam)
        
        return cams
    
    def generate_multi_layer(self, input_tensor, layers, target_class=None):
        """Generate Grad-CAM from multiple layers and combine."""
        import numpy as np
        
        cams = []
        for layer in layers:
            self.target_layer = layer
            self._register_hooks()
            cam = self.generate(input_tensor, target_class)
            cams.append(cam)
        
        # Average CAMs from different layers
        combined_cam = np.mean(cams, axis=0)
        combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
        return combined_cam

class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ for improved localization."""
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Grad-CAM++ weighting
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha weights
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        sum_activations = activations.sum(dim=[2, 3], keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        weights = (alpha * F.relu(gradients)).sum(dim=[2, 3], keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
