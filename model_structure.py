import torch
import torchvision
from torch import nn
import os
import glob
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, z: int,
                 img_size: int,
                 img_channel: int,
                 ):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z,
                               out_channels=img_size*16,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=img_size*16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=img_size*16,
                               out_channels=img_size*8,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=img_size*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=img_size*8,
                               out_channels=img_size*4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=img_size*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=img_size*4,
                               out_channels=img_size*2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=img_size*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=img_size*2,
                               out_channels=img_size,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                               ),
            nn.BatchNorm2d(num_features=img_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=img_size,
                               out_channels=img_channel,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.generator(x)
    
class Discriminator(nn.Module):
    def __init__(self,
                 img_size: int,
                 img_channel: int
                ):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=img_channel,
                      out_channels=img_size,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=img_size,
                      out_channels=img_size*2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=img_size*2,
                      out_channels=img_size*4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=img_size*4,
                      out_channels=img_size*8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=img_size*8,
                      out_channels=img_size*16,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=img_size*16,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=0,
                      bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)
