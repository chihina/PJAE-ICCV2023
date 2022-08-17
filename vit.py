import  numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import sys
import timm
import math
from einops.layers.torch import Rearrange

image_height = 320
image_width = 480

class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        patch_height = 4
        patch_width = 4

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 3 * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, 64),
        )


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        return x

model = ViT()
img = torch.zeros(2, 3, image_height, image_width)
print(img.shape)
out = model(img)
print(out.shape)