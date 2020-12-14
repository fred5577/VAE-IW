import glob
import os
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class AtariDataloader(torch.utils.data.Dataset):
    """
    Description of fishy class
    Attributes:
        train : Percentage of set used for training.
        transform : 
        data_path : Path to images
    """

    def __init__(self, train, transform, data_path):
        """
        Constructor for Fishy class
            Parameters:
                train : Percentage of set used for training.
                transform : 
                data_path : Path to images
        """
        self.transform = transform
        data_path = os.path.join(data_path)
        
        #self.name_to_label = [i.split(";")[1][1:-1] for i in fp]
        self.image_paths = glob.glob(data_path + '/*.png')
        
    def __len__(self):
        """
        Returns the total number of samples
        Returns :
            int : The total number of images
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Generates one sample of data
        Parameters:
            idx (int): Index for image
        Returns :
            Image : Transformed image.
        """
        image_path = self.image_paths[idx]
        
        lookup = image_path.split("/")[-1].split(".")[0]
        
        image = Image.open(image_path)
        X = self.transform(image)
        return X


def createDataLoaders(batch_size, data_path='../Pictures/SpaceInvaders-v4', datasetsize=(128,128), train_distribution = 0.95, gray=True, **kwargs):
    #For testing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(datasetsize),
        transforms.ToTensor()]
    )

    if not gray:
        transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(datasetsize),
            transforms.ToTensor()]
        )

    full_dataset = AtariDataloader(train=True, transform=transform,data_path=data_path)
    train_size = int(train_distribution * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, trainset, testset

# train_loader, test_loader, trainset, testset = createDataLoaders(32)

# batch_size = 32
# transform = transforms.Compose(
#     [transforms.ToTensor()]
# )

# trainset = SpaceInvaders(train=True, transform=transform)
# testset = SpaceInvaders(train=False, transform=transform)

# print(len(trainset))