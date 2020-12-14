import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Bernoulli, kl_divergence
from vae.utils import CropImage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2, leaky=0.01, downsample=None):
        super(ResidualBlock, self).__init__()
        # TODO: Maybe change order of res block from, conv->acti->batch->drop
        # to batch->acti->conv->drop
        self.leaky = leaky
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x):
        residual = x
        x = self.block(x)
        x = F.leaky_relu(x,self.leaky)
        return x

class Dif_ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2, leaky=0.01, downsample=None):
        super(Dif_ResidualBlock, self).__init__()
        # TODO: Maybe change order of res block from, conv->acti->batch->drop
        # to batch->acti->conv->drop
        self.leaky = leaky
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(self.leaky),
            nn.Conv2d(channels, channels, kernel_size=3,padding=1),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x):
        residual = x
        x = self.block(x)
        return x


class Res_3d_Conv_15(nn.Module):
    def __init__(self, latent, block, image_size, temp, image_channels=1):
        super(Res_3d_Conv_15, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_out_size = 15
        self.image_size = image_size
        self.pz = Bernoulli(probs=torch.tensor(.5))
        channels = 64
        self.image_channels = image_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.image_channels, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2), 
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent, channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, self.image_channels, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )
       

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, z):
        z = self.decoder(z)
        return z

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample([n_imgs, self.latent,self.image_out_size,self.image_out_size]).to(device)
        mean = self.decode(z)
        pxz = Normal(mean, 1)
        return mean  # pxz.sample()
    
    def bernoulli(self, x):
        q = self.encode(x)
        qz = Bernoulli(logits=q)
        return qz

    # TODO: Implement a methods that uses argmax instead of softmax activation that can be used when the model is trained
    def get_features(self, x):
        q = self.encode(x)
        padded_logits = q.unsqueeze(-1)
        padded_logits = torch.cat(
                [padded_logits, torch.zeros_like(padded_logits)], dim=-1)
        z = F.gumbel_softmax(padded_logits,0.5,hard=False)
        z = z[:,:,:,:,0]
        z = z.view(-1, self.latent*self.image_out_size*self.image_out_size)
        return z

    def forward(self, x, temp, hard):
        q = self.encode(x)
        q = q.clamp(min=-8., max=8.)
        padded_logits = q.unsqueeze(-1)
        padded_logits = torch.cat(
                [padded_logits, torch.zeros_like(padded_logits)], dim=-1)
        pz = F.gumbel_softmax(padded_logits, temp, hard=hard)
        qz = Bernoulli(logits=q)
        z = pz[:,:,:,:,0]
        return self.decode(z), qz, pz

