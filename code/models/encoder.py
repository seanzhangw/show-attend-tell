import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=7):
        super().__init__()

        # Load pretrained ResNet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove avgpool and fc
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Freeze CNN weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

    def forward(self, images):
        """
        images: (B, 3, 224, 224)

        returns:
            (B, num_pixels=49, feature_dim=2048)
        """
        # (B, 2048, H, W), is (B, 2048, 7, 7) for 224 x 224 images
        features = self.resnet(images)

        # ensure fixed size (B, 2048, 7, 7) if images are not 224 x 224
        features = self.adaptive_pool(features)

        # reshape for attention
        B, C, _, _ = features.size() # (batch size, channels, height, width)

        features = features.view(B, C, -1)       # (B, 2048, 49)
        features = features.permute(0, 2, 1)     # (B, 49, 2048)

        return features