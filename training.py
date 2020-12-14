import argparse
import numpy as np
import os
import vae.Loader as Loader
import sys
import math
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vae.utils import CropImage, linear_schedule
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Bernoulli, kl_divergence
from vae.models import ResidualBlock, Dif_ResidualBlock, Res_3d_Conv_15

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=0.5, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')
parser.add_argument('--annealing', type=bool, default=False,
                    help='Temperature annealing')
parser.add_argument('--env', default="SpaceInvaders",
                    help='Name of environment to test for.')
parser.add_argument('--zdim', type=int,  default=1000,
                    help='the value of the last dimension of the latent space.')
parser.add_argument('--file-naming', default="",
                    help='the value of the last dimension of the latent space.')
parser.add_argument('--beta', type=float, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--sigma', type=float, default=1, metavar='N',
                    help='value of gaussian for loss (default: 1)')
parser.add_argument('--loss', default="MSE", metavar='N',
                    help='Coosing the loss to be used')
parser.add_argument('--image-training-size', type=int, default=84, metavar='N',
                    help='Coosing the loss to be used')
parser.add_argument('--kernel-size',default="15", metavar='N',
                    help='Choose kernel size')
parser.add_argument('--image-channels',type=int, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

env = args.env
height=args.image_training_size
width=args.image_training_size
image_pixels = height * width
kernel_size=args.kernel_size
channels = 64
image_channels = args.image_channels
z_dim = args.zdim

recons_path = "data/{}/recons/sigma_{}_beta_{}_zdim_{}_{}_{}_imagesize_{}_k_{}_temp_{}_A_{}_{}".format(env, args.sigma, args.beta, z_dim,channels,args.loss, args.image_training_size, kernel_size,args.temp,args.annealing, args.file_naming)
if not os.path.exists(recons_path):
    os.makedirs(recons_path)
recons_path = recons_path + '/reconstruction_'

sample_path = "data/{}/sample/sigma_{}_beta_{}_zdim_{}_{}_{}_imagesize_{}_k_{}_temp_{}_A_{}_{}".format(env, args.sigma, args.beta, z_dim,channels,args.loss, args.image_training_size,kernel_size,args.temp,args.annealing, args.file_naming)
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
sample_path = sample_path + '/sample_'

if not os.path.exists("data/{}/model".format(env)):
    os.makedirs("data/{}/model".format(env))
model_path = "data/{}/model/model_res_sigma_{}_beta_{}_zdim_{}_{}_{}_imagesize_{}_k_{}_temp_{}_A_{}_{}".format(env, args.sigma, args.beta,z_dim,channels,args.loss, args.image_training_size,kernel_size,args.temp,args.annealing, args.file_naming)

if not os.path.exists("data/{}/plots".format(env)):
    os.makedirs("data/{}/plots".format(env))
histo_path = "data/{}/plots/sigma_{}_beta_{}_zdim_{}_{}_{}_imagesize_{}_k_{}_temp_{}_A_{}_{}".format(env,args.sigma, args.beta,z_dim,channels,args.loss, args.image_training_size,kernel_size,args.temp,args.annealing, args.file_naming)
print("path for model:", model_path)



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Env: " , env, " Epochs: " , args.epochs, " Batch size: ", args.batch_size , " Temperature: " , args.temp, " Cuda: " 
        , device, " Annealing: " , args.annealing, "Sigma: ", args.sigma, " Beta: ", args.beta, " Zdim: ", z_dim , 
        " Image trainign size: ", args.image_training_size)


kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader, trainset, testset = Loader.createDataLoaders(args.batch_size, data_path='../Pictures-v1/{0}-v4'.format(env), datasetsize=(args.image_training_size,args.image_training_size), gray=True, **kwargs)

temp_min = 0.5
ANNEAL_RATE = math.log(args.temp/temp_min) + 2

print(kernel_size)
model = Res_3d_Conv_15(z_dim, ResidualBlock, args.image_training_size, args.temp,image_channels).to(device)


print(model)
#optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

points_loss_train = []
points_kld_train = []
points_recons_train = []
points_mse_train = []
points_x_train = []
 
test_loss_train = []
test_kld_train = []
test_recons_train = []
test_mse_train = []
test_x_train = []
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, z, epoch, choose_loss = "MSE", beta=args.beta, sigma=args.sigma):
    # Loss from article on SAE
    if choose_loss == "BCE":
        # You cant interpret this

        # Binary cross entropy loss (negative log likelihood)
        BCE = F.binary_cross_entropy(recon_x.view(-1,image_pixels), x.view(-1, image_pixels),reduction='sum') / x.shape[0]

        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()

        # Tries to avoid constant flipping of coins
        zerosuppress_loss = torch.mean(z) * linear_schedule(0.7, epoch)
        return BCE + KLD*beta + zerosuppress_loss, BCE, KLD, F.mse_loss(recon_x,x) 
    elif choose_loss == "MSE":
        # KL-divergence
        pz = Bernoulli(probs=torch.tensor(.5))
        KLD = kl_divergence(qy, pz).sum((1,2,3)).mean()
        #log_ratio = torch.log(z * 2 + 1e-20)
        #KLD = (log_ratio * z).sum((1,2,3)).mean()

        # Gaussian log likelihood
        pxz = Normal(recon_x, sigma)
        log_likelihood = -pxz.log_prob(x).sum((1,2,3)).mean()

        # Zero suppress
        zerosuppress_loss = torch.mean(z) * linear_schedule(0.7, epoch)
        return log_likelihood + KLD*beta, log_likelihood, KLD, F.mse_loss(recon_x,x)


def train(epoch):
    model.train()

    train_loss = 0
    kld_loss = 0
    reconstruction_loss = 0
    temp = args.temp
    # if args.annealing:
    #     temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        step = batch_idx * len(data) + (epoch-1)*len(train_loader.dataset)
        if args.annealing and step % 15000*0.95*25-26  == 0 and step % (15000*0.95*50-26-26) == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * step), temp_min)

        optimizer.zero_grad()
        recon_batch, qz, z = model(data, temp, args.hard)
    
        loss, recons, kld, mse = loss_function(recon_batch, data, qz, z, epoch, choose_loss=args.loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        train_loss += loss.item() * len(data)
        kld_loss += kld.item() * len(data)
        reconstruction_loss += recons.item() * len(data)
        optimizer.step()
        
        points_loss_train.append(loss.item())
        points_kld_train.append(kld.item())
        points_recons_train.append(recons.item())
        points_mse_train.append(math.log(mse.item()))
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
    temp = args.temp
    # if args.annealing:
    #     temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
    #temp = np.maximum(args.temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
    for i, data in enumerate(test_loader):
        step = i * len(data) + (epoch-1)*len(test_loader.dataset)
        
        if args.cuda:
            data = data.to(device)
        if args.annealing and step % 960  == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * step), temp_min)

        recon_batch, qz, z = model(data, temp, args.hard)
        loss, recons, kld, mse = loss_function(recon_batch, data, qz, z, epoch)
        test_loss += loss.item() * len(data)
        kld_loss += kld.item() * len(data)
        reconstruction_loss += recons.item() * len(data)

        test_loss_train.append(loss.item())
        test_kld_train.append(kld.item())
        test_recons_train.append(recons.item())
        test_mse_train.append(math.log(mse.item()))
        test_x_train.append(step)
        # if i == 0:
        #     print("Models latent variance noise over 10 samples from test data", model.latent_variance_noise(data))
        
        if i == 0 and epoch % 10 == 0:
            # points_loss_test.append(loss.item())
            # points_kld_test.append(kld.item())
            # points_recons_test.append(recons.item())
            # points_x_test.append(epoch*batch_idx * len(data))
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