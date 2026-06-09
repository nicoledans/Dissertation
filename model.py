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

        # Keep layer-4 feature maps and gradients for Grad-CAM evaluation.
        self._forward_hook = self.backbone.layer4.register_forward_hook(self._save_activations)
        self._backward_hook = self.backbone.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def class_scores(logits, labels=None):
        """Return the score for the relevant binary class.

        With labels, target the ground-truth class for explanation-supervised
        training. Without labels, target the model's predicted class for
        post-hoc explanation.
        """
        if labels is None:
            signs = torch.where(logits.detach() >= 0, 1.0, -1.0)
        else:
            signs = labels.to(dtype=logits.dtype) * 2.0 - 1.0
        return logits * signs

    @property
    def activations(self):
        if self._activations is None:
            raise RuntimeError("No activations available. Run a forward pass first.")
        return self._activations

    @staticmethod
    def normalise_gradcam(cam):
        """Normalise each CAM independently to [0, 1]."""
        b = cam.shape[0]
        cam_flat = cam.flatten(start_dim=1)
        cam_min = cam_flat.min(dim=1).values.view(b, 1, 1)
        cam_max = cam_flat.max(dim=1).values.view(b, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def get_gradcam(self, normalise=True):
        """Return post-hoc Grad-CAM after backward() has populated gradients."""
        if self._gradients is None:
            raise RuntimeError("No gradients available. Run backward() before get_gradcam().")
        grads = self._gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))
        return self.normalise_gradcam(cam) if normalise else cam

    def differentiable_gradcam(self, scores, normalise=True):
        """Return class-specific Grad-CAM while retaining gradients for training.

        This requires second-order gradients and is therefore more expensive
        than ordinary classification training.
        """
        acts = self.activations
        grads = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=acts,
            create_graph=True,
            retain_graph=True,
        )[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))
        return self.normalise_gradcam(cam) if normalise else cam

    def clear_hooks(self):
        """Clear stored hook tensors without removing the registered hooks."""
        self._activations = None
        self._gradients = None

    def remove_hooks(self):
        """Remove registered hooks when the model is no longer needed."""
        self._forward_hook.remove()
        self._backward_hook.remove()
