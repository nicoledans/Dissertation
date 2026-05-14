import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class NoduleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace classification head: Dropout->Linear(2048,256)->ReLU->Linear(256,1)
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.backbone = backbone

        self._activations = None
        self._gradients = None

        # Forward hook layer4 - store feature maps
        self.backbone.layer4.register_forward_hook(self._save_activations)
        # Backward hook layer4 -  store gradients
        self.backbone.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def forward(self, x):
        return self.backbone(x)

    def get_gradcam(self):
        grads = self._gradients          # (B, C, H, W)
        acts = self._activations         # (B, C, H, W)

        # Global average pool gradients to get per-channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1)               # (B, H, W)
        cam = torch.relu(cam)

        # Normalise each map in [0, 1]
        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1)[0].view(b, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(b, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam  # (B, H, W) tensor, values in [0,1]

    def clear_hooks(self):
        self._activations = None
        self._gradients = None
