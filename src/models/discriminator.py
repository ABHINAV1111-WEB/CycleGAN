"""The discriminator learns to distinguish between real images 
and generated (fake) images. CycleGAN uses PatchGAN discriminators, 
which classify image 
patches rather than the whole image— making it more efficient 
and better at capturing texture

# The discriminator doesn’t say “this whole image is fake.”
 Instead, it says “this patch looks fake” — which forces the generator to improve
local realism (textures, edges, details).

"""

import torch
import torch.nn as nn

# Defining the discriminator class
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3, norm_layer= nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        layers= []

        # Initial layer (no normalization)
        layers += [nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, inplace= True)]
        
        # Intermediate layers
        nf_mult= 1
        for n in range(1, n_layers):
            nf_mult_prev= nf_mult
            nf_mult= min(2 **n, 8)
            layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                 kernel_size=4, stride=2, padding=1, bias= False),
                                 norm_layer(ndf * nf_mult),
                                 nn.LeakyReLU(0.2, inplace=True)
                                 ]
            
        # Final layer

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size = 4, stride=1, padding= 1, bias= False),
                             norm_layer(ndf* nf_mult),
                             nn.LeakyReLU(0.2, inplace= True)]
        
        # Output layer: 1-channel prediction map
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Alias for backward compatibility
Discriminator = NLayerDiscriminator