import argparse
import numpy as np
import os
import vae.Loader as Loader
import sys
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vae.utils import CropImage, linear_schedule, data_dependent_init
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Bernoulli, kl_divergence
from vae.models_norm import ResidualBlock, Norm_3d_Conv_15

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--env', default="SpaceInvaders",
                    help='Name of environment to test for.')
parser.add_argument('--latent-out', type=int,  default=500,
                    help='the value of the last dimension of the latent space.')
parser.add_argument('--file-naming', default="",
                    help='the value of the last dimension of the latent space.')
parser.add_argument('--beta', type=float, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--loss', default="BCE", metavar='N',
                    help='Coosing the loss to be used')
parser.add_argument('--image-training-size', type=int, default=128, metavar='N',
                    help='Coosing the loss to be used')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

env = args.env
height=args.image_training_size
width=args.image_training_size
image_pixels = height * width
channels = 128
image_channels = 1
latent_out = args.latent_out

if not os.path.exists("data/{}/gaus/recons/beta_{}_zdim_{}_{}_imagesize_{}_{}".format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming)):
    os.makedirs("data/{}/gaus/recons/beta_{}_zdim_{}_{}_imagesize_{}_{}".format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming))
recons_path = 'data/{}/gaus/recons/beta_{}_zdim_{}_{}_imagesize_{}_{}/reconstruction_'.format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming)
if not os.path.exists("data/{}/gaus/sample/beta_{}_zdim_{}_{}_imagesize_{}_{}".format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming)):
    os.makedirs("data/{}/gaus/sample/beta_{}_zdim_{}_{}_imagesize_{}_{}".format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming))
sample_path = 'data/{}/gaus/sample/beta_{}_zdim_{}_{}_imagesize_{}_{}/sample_'.format(env, args.beta, latent_out,channels,args.loss, args.image_training_size, args.file_naming)

model_path = "data/{}/gaus/model/model_res_beta_{}_zdim_{}_{}_imagesize_{}_{}".format(env, args.beta,latent_out,channels,args.loss, args.image_training_size, args.file_naming)
histo_path = "data/{}/gaus/plots/beta_{}_zdim_{}_imagesize_{}_{}".format(env, args.beta,latent_out,channels,args.loss, args.image_training_size, args.file_naming)
print("path for model:", model_path)



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Env: " , env, " Epochs: " , args.epochs, " Batch size: ", args.batch_size , " Cuda: " 
        , device, " Beta: ", args.beta, " latent out: ", latent_out , 
        " Image trainign size: ", args.image_training_size)


kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader, trainset, testset = Loader.createDataLoaders(args.batch_size, data_path='../Pictures/{0}-v4'.format(env), datasetsize=(args.image_training_size,args.image_training_size), **kwargs)

#model = Res_3d_Conv_4(z_dim, ResidualBlock, args.image_training_size, args.temp).to(device)
model = Norm_3d_Conv_15(latent_out, ResidualBlock, 128).to(device)
# Get batch
t = []
for batch_idx, data in enumerate(train_loader):
    t = data.to(device)
    break
# Use batch for data dependent init
data_dependent_init(model, {'x': t.to(device)})
print(model)
#optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

points_loss_train = []
points_kld_train = []
points_recons_train = []
points_x_train = []
 
test_loss_train = []
test_kld_train = []
test_recons_train = []
test_x_train = []
# Reconstruction + KL divergence losses summed over all elements and batch
reconstruction_function = nn.MSELoss(reduction="sum")
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1,image_pixels), x.view(-1, image_pixels),reduction='sum') / x.shape[0]
    # reconstruction_function(recon_x,x)#
    #BCE = BCE * 0.5
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    qy = Normal(mu,logvar)
    pz = Normal(torch.tensor([0]).to(device), torch.tensor([1]).to(device))
    KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()
    #KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1), dim = 0)
    # KLD = torch.sum(kld_loss)

    return BCE + KLD*args.beta,BCE,KLD

def train(epoch):
    model.train()

    train_loss = 0
    kld_loss = 0
    reconstruction_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        step = batch_idx * len(data) + (epoch-1)*len(train_loader.dataset)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss, recons, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        train_loss += loss.item() * len(data)
        kld_loss += kld.item() * len(data)
        reconstruction_loss += recons.item() * len(data)
        optimizer.step()
        
        points_loss_train.append(loss.item())
        points_kld_train.append(kld.item())
        points_recons_train.append(recons.item())
        points_x_train.append(step)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, KLD loss: {}, Reconstruction loss: {}'.format(
        epoch, train_loss / len(train_loader.dataset), kld_loss / len(train_loader.dataset), reconstruction_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    kld_loss = 0
    reconstruction_loss = 0
    for i, data in enumerate(test_loader):
        step = i * len(data) + (epoch-1)*len(test_loader.dataset)
        
        if args.cuda:
            data = data.to(device)

        recon_batch, mu, logvar = model(data)
        loss, recons, kld = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.item() * len(data)
        kld_loss += kld.item() * len(data)
        reconstruction_loss += recons.item() * len(data)

        test_loss_train.append(loss.item())
        test_kld_train.append(kld.item())
        test_recons_train.append(recons.item())
        test_x_train.append(step)

        if i == 0 and epoch % 10 == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, image_channels, height, width)[:n]])
            save_image(comparison.data.cpu(), recons_path + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    reconstruction_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}, kld loss: {}, reconstruction loss: {}'.format(test_loss, kld_loss, reconstruction_loss))


def run():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            sample = model.sample_prior(16)
            if args.cuda:
                sample = sample.cuda()
            #sample = model.decode(sample).cpu()
            save_image(sample.data.view(16, image_channels, height, width), sample_path + str(epoch) + '.png')
       
        if epoch == 100:
            create_loss_plot(epoch)
            torch.save(model.state_dict(), model_path + "_epochs_" + str(epoch) + ".pt")
        elif epoch == 150:
            create_loss_plot(epoch)
            torch.save(model.state_dict(),  model_path + "_epochs_" + str(epoch) + ".pt")
        elif epoch == 200:
            create_loss_plot(epoch)
            torch.save(model.state_dict(), model_path + "_epochs_" + str(epoch) + ".pt")


def create_loss_plot(epoch):
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    axes[0, 0].plot(points_x_train,points_loss_train)
    axes[0, 0].set_title("loss")
    axes[0, 1].plot(points_x_train, points_kld_train)
    axes[0, 1].set_title("kld")
    axes[1, 0].plot(points_x_train, points_recons_train)
    axes[1, 0].set_title("recons")
    figure.savefig(histo_path + "_epcoh_" + str(epoch) + "_plot_train.png")
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    axes[0, 0].plot(test_x_train,test_loss_train)
    axes[0, 0].set_title("loss")
    axes[0, 1].plot(test_x_train, test_kld_train)
    axes[0, 1].set_title("kld")
    axes[1, 0].plot(test_x_train, test_recons_train)
    axes[1, 0].set_title("recons")
    figure.savefig(histo_path + "_epcoh_" + str(epoch) + "_plot_test.png")



# TODO: implement a method that can evaluate the performances of the VAE, since looking at images is not the best indicator.
if __name__ == '__main__':
    run()