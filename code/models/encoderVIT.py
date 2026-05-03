import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class EncoderViT(nn.Module):
    def __init__(self):
        super(EncoderViT, self).__init__()
        # Load the pre-trained ViT (Base model, 16x16 patches)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Freeze the ViT weights so we only train the Decoder/Attention!
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, images):
        x = self.vit._process_input(images)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = self.vit.encoder(x) 
        
        # x is now shape: (Batch_Size, 197, 768)
        patches = x[:, 1:, :] 
        # patches is now shape: (Batch_Size, 196, 768)
        return patches