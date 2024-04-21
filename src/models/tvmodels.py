# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.models import ViT_B_16_Weights
from transformers import ViTForImageClassification


__all__ = ["mobilenet_v3_small", "vgg16", "vit_b_16", "ViT"]

class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()
        self.loss = loss
        self.is_transformer = name in ["vit_b_16", "ViT"]  
        if self.is_transformer:
            self.backbone = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Replace classifier for feature extraction
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        if self.is_transformer:
            outputs = self.backbone(x)
            # Extract features directly from the logits (as we've set the original classifier to Identity)
            features = outputs.logits
        else:
            features = self.backbone(x)

        if not self.training:
            return features  # Return features directly during evaluation

        logits = self.classifier(features)
        if self.loss == {"xent"}:
            return logits
        elif self.loss == {"xent", "htri"}:
            return logits, features
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")




def vgg16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vgg16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html

def vit_b_16(num_classes, loss = {"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vit_b_16",
        num_classes=num_classes,
        loss = loss,
        pretrained=pretrained,
        **kwargs,  
    )
    return model

def ViT(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
