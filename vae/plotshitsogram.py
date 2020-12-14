import argparse
import numpy as np
import os
import Loader as Loader
import sys
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import CropImage, linear_schedule
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Bernoulli, kl_divergence
from models import Res_3d_Conv_7, ResidualBlock, Dif_ResidualBlock, Res_3d_Conv_4, Res_3d_Conv_15, Res_3d_Conv_32,Res_3d_Conv_1, Res_3d_Conv_16,Res_3d_Conv_2, Res_3d_Conv_15_big


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor()]
)
print("Loading model")
stat_dictionary = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Res_3d_Conv_15(20, ResidualBlock, 128, 0.5)
domain="Alien"
model.load_state_dict(torch.load("../data/"+ domain + "/model/model_res_sigma_1.0_beta_0.0001_zdim_20_64_BCE_imagesize_128_k_15_temp_0.5_A_False_neg_kl_epochs_100.pt",map_location=device))
model.eval()
model.to(device)
print("Done with loading model")

kwargs = {'num_workers': 6, 'pin_memory': True}

train_loader, test_loader, trainset, testset = Loader.createDataLoaders(64, data_path='../../Pictures/' + domain + '-v4', datasetsize=(128,128), **kwargs)

points = np.array([])
model.eval()
counter = 0
for i, data in enumerate(test_loader):
    data = data.to(device)
    recon_batch, qz, z = model(data, 0.5, False)
    
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n],
                            recon_batch.view(64, 1, 128, 128)[:n]])
    save_image(comparison.data.cpu(), 'recons_images/test.pdf', nrow=n)
    break

# fig  = plt.hist(points, bins = 10)
# plt.savefig("hist_" + domain + "_test.png")