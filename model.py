import torch
import torch.nn as nn
import torchvision.models as models


class NoduleClassifier(nn.Module):
    """Shared CT-slice classifier used by the baseline and later experiments.

    A neural network is a large collection of learnable numbers called
    parameters. During training, the loss measures how wrong the prediction is,
    and an optimizer changes these parameters to reduce future errors.

    This model has two main parts:

    1. A ResNet-50 "backbone" that converts an image into useful visual features.
    2. A custom classification "head" that converts those features into one
       benign-versus-malignant score.

    Grad-CAM hooks are also attached to the final convolutional stage. Hooks let
    us observe intermediate feature maps and their gradients without changing
    the normal forward prediction.
    """

    def __init__(self):
        super().__init__()

        # ResNet-50 is a deep convolutional neural network. Convolutions scan
        # learned filters across an image to detect patterns such as edges,
        # textures, and increasingly complex structures.
        #
        # "weights=IMAGENET1K_V1" means the backbone does not begin with random
        # parameters. It begins with parameters learned from the large ImageNet
        # natural-image dataset. CT images are different from ImageNet images,
        # but pretrained edge and texture detectors are usually a better starting
        # point than completely random filters, especially for a small dataset.
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Standard ResNet-50 ends with a layer that predicts 1,000 ImageNet
        # object classes. We replace that layer because this project needs one
        # binary malignancy score instead.
        #
        # ResNet's preceding layers finish with 2,048 extracted features.
        #
        # Dropout(0.5):
        #   During TRAINING only, randomly hides approximately half of these
        #   features on each forward pass. This discourages the classifier from
        #   depending too heavily on a few features and can reduce memorisation.
        #   During validation/testing, Dropout is automatically disabled.
        #
        # Linear(2048, 256):
        #   A Linear layer learns weighted combinations of its inputs. This one
        #   compresses the 2,048 ResNet features into 256 task-specific features.
        #
        # ReLU:
        #   Replaces negative values with zero while keeping positive values.
        #   This nonlinearity lets stacked layers learn complicated relationships;
        #   without it, multiple Linear layers would collapse into one simple
        #   Linear transformation.
        #
        # Linear(256, 1):
        #   Combines the 256 features into one raw score called a logit.
        #   A negative logit leans benign; a positive logit leans malignant.
        #   BCEWithLogitsLoss later converts this logit internally using sigmoid.
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.backbone = backbone

        # These variables temporarily hold the information needed for Grad-CAM.
        # Activations are the feature maps produced by layer4.
        # Gradients describe how strongly those feature maps affect a class score.
        self._activations = None
        self._gradients = None

        # A hook is an observer attached to a layer. It records information when
        # data passes through that layer, but does not alter the prediction.
        #
        # Grad-CAM uses layer4 because it is ResNet's final convolutional stage:
        # its features are semantically rich while still retaining some spatial
        # layout. Later average pooling removes that spatial layout.
        self._forward_hook = self.backbone.layer4.register_forward_hook(self._save_activations)
        self._backward_hook = self.backbone.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Remember layer4's forward feature maps for a later Grad-CAM."""
        self._activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        """Remember how the selected class score depends on layer4 features."""
        self._gradients = grad_output[0]

    def forward(self, x):
        """Run a batch of images through ResNet and return one logit per image."""
        return self.backbone(x)

    @staticmethod
    def class_scores(logits, labels=None):
        """Return the score for the relevant binary class.

        Binary classification has one logit rather than separate benign and
        malignant outputs:

        - A positive logit supports malignant.
        - A negative logit supports benign.

        Grad-CAM needs a score to explain. Multiplying by +1 explains malignant;
        multiplying by -1 explains benign.

        With labels, explain the correct/ground-truth class. Without labels,
        explain whichever class the model currently predicts.
        """
        if labels is None:
            # detach() makes the class choice a fixed decision for this Grad-CAM;
            # training cannot manipulate the >= 0 comparison itself.
            signs = torch.where(logits.detach() >= 0, 1.0, -1.0)
        else:
            # Convert benign label 0 to sign -1 and malignant label 1 to sign +1.
            signs = labels.to(dtype=logits.dtype) * 2.0 - 1.0
        return logits * signs

    @property
    def activations(self):
        if self._activations is None:
            raise RuntimeError("No activations available. Run a forward pass first.")
        return self._activations

    @staticmethod
    def normalise_gradcam(cam):
        """Scale every sample's heatmap independently into the range [0, 1].

        This makes 0 mean the least-active location in that heatmap and 1 mean
        the most-active location. Independent scaling makes visual comparisons
        easier, but it means Grad-CAM values are relative within each image.
        """
        b = cam.shape[0]
        cam_flat = cam.flatten(start_dim=1)
        cam_min = cam_flat.min(dim=1).values.view(b, 1, 1)
        cam_max = cam_flat.max(dim=1).values.view(b, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def get_gradcam(self, normalise=True):
        """Return an ordinary post-hoc Grad-CAM after a backward pass.

        Grad-CAM asks:
        "Which spatial regions in layer4 most supported this class score?"

        It averages gradients spatially to obtain one importance weight per
        feature channel, combines the weighted feature maps, then applies ReLU
        so only positive support for the selected class remains.

        This version is used for evaluation/visualisation after prediction.
        """
        if self._gradients is None:
            raise RuntimeError("No gradients available. Run backward() before get_gradcam().")
        grads = self._gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))
        return self.normalise_gradcam(cam) if normalise else cam

    def differentiable_gradcam(self, scores, normalise=True):
        """Return class-specific Grad-CAM while retaining gradients for training.

        Ordinary Grad-CAM explains a completed prediction. Some experiments in
        this project also put Grad-CAM inside the training loss, allowing the
        optimizer to change the model based on where it looked.

        Training through Grad-CAM requires gradients of gradients, called
        second-order gradients. This is why attention-supervised experiments are
        slower and use more memory than the classification-only baseline.
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
        """Forget tensors from the previous prediction while keeping hooks active."""
        self._activations = None
        self._gradients = None

    def remove_hooks(self):
        """Detach Grad-CAM observers when the model is no longer needed."""
        self._forward_hook.remove()
        self._backward_hook.remove()
