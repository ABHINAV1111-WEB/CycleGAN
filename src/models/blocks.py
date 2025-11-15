"""🧱 Purpose of blocks.py
This file defines the building blocks used inside your generator model — like LEGO pieces that help the model learn better image transformations.

🎯 What does it contain?
It defines a special unit called a ResNet block, which is used to:
- Learn small changes to the image (called residuals)
- Preserve important features while transforming style
- Make the generator deeper without breaking training

- It use a trick called a skip connection — which means the input is added back to the output. This helps the model remember what’s important.

"""

import torch
import torch.nn as nn

# Resnnet Block : learns small changes while preserving structure
# InstancNorm2d: Stablizes training for style transfer
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """A single ResNet block with reflection padding."""
        super(ResnetBlock, self).__init__()
        block = []

        # First conv, ReflectionPad2d: Reduces edge artifacts in images
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
                  norm_layer(dim),
                  nn.ReLU(True)]
        # Optional dropout
        if use_dropout:
            block += [nn.Dropout(0.5)]
        # second conv
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
                  norm_layer(dim)]
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)  # Residual connection