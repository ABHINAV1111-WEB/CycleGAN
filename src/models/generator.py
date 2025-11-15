
import torch
import torch.nn as nn
from .blocks import ResnetBlock

class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9, norm_layer= nn.InstanceNorm2d, use_reflection= True, upsample= "deconv" ):
        """
        Args:
            in_channesl: Number of input channels(e.g., 3 for RGB)
            out_channels: Number of output channels
            ngf: Base numer of residual blocks
            norm_layer: Normalization layer (InstanceNorm2d)
            use_reflection: Whether to use feflection padding
            upsample: "deconv" or "nearest+conv"
                        
        """
        super(ResnetGenerator, self).__init__()
        model= []

        # Initial conv
        if use_reflection:
            model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias= False),
                norm_layer(ngf),
                  nn.ReLU(True)  ]
        
        # Downsampling
        n_down=2
        for i in range(n_down):
            mult= 2**i
            model +=[nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3,stride= 2, padding=1, bias= False),
                     norm_layer(ngf*mult*2),
                     nn.ReLU(True)
                      ]
            
        # Residual blocks
        mult= 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]
        
        # Upsampling
        for i in range(n_down):
            mult = 2**(n_down - 1 - i)
            if upsample == "deconv":
                model += [nn.ConvTranspose2d(int(ngf*mult*2), int(ngf*mult),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1, bias=False),
                          norm_layer(int(ngf*mult)),
                          nn.ReLU(True)]
                
        # Final conv
        if use_reflection:
            model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                  nn.Tanh()
                  ]
            
        self.model= nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Alias for backward compatibility
Generator = ResnetGenerator