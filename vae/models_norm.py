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
from torch.autograd import Variable

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



class Norm_3d_15(nn.Module):
    def __init__(self, latent, block, image_size,latent_variable_size):
        super(Norm_3d_15, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_size = image_size
        self.latent_variable_size = latent_variable_size
        channels = 64
        self.pz = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2), 
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.fc1 = nn.Linear(15*15*20, latent_variable_size)
        self.fc2 = nn.Linear(15*15*20, latent_variable_size)

        self.d1 = nn.Linear(latent_variable_size, 15*15*20)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent, channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, 1, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )

        self.leakyrelu = nn.LeakyReLU(leaky)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 15*15*20)
        return self.fc1(x), self.fc2(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 20, 15, 15)
        h1 = self.decoder(h1)
        return h1

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample(torch.tensor([n_imgs, self.latent_variable_size])).to(device)
        z = z.view(n_imgs,self.latent_variable_size)
        mean = self.decode(z)
        return mean  # pxz.sample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

class Norm_3d_Conv_15(nn.Module):
    def __init__(self, latent, block, image_size):
        super(Norm_3d_Conv_15, self).__init__()
        self.latent = latent
        dropout = 0.2
        leaky = 0.01
        self.image_out_size = 15
        self.image_size = image_size
        channels = 64
        self.pz = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2), 
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.Conv2d(channels, self.latent, kernel_size=3, padding=1, stride=2), #4x4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d((int)(self.latent/2), channels, 3, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, channels, 4, stride=2),
            nn.LeakyReLU(leaky),
            block(channels, dropout=dropout),
            nn.ConvTranspose2d(channels, 1, 4, stride=2),  # 86x86
            CropImage((image_size, image_size)),
            nn.Sigmoid(),
        )

        self.leakyrelu = nn.LeakyReLU(leaky)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       

    def encode(self, x):
        x = self.encoder(x)
        mu,logvar = torch.chunk(x,2,dim=1)
        return mu,logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z

    def get_features(self, x):
        mu, logvar = self.encode(x)
        #z = self.reparametrize(mu, logvar)
        return mu.view(-1, (int)(self.latent/2) * self.image_out_size * self.image_out_size)

    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample(torch.tensor([n_imgs, (int)(self.latent/2),self.image_out_size, self.image_out_size])).to(device)
        z = z.view(n_imgs,(int)(self.latent/2),self.image_out_size, self.image_out_size)
        mean = self.decode(z)
        return mean  # pxz.sample()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar
