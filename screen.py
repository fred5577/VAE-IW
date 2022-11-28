import numpy as np
import time
import traceback
import sys#, platform
# import cProfile, pstats, io
# from pstats import SortKey
import bprost as bprost_lib
import torch
from scipy.stats import norm
from vae.models import ResidualBlock, Res_3d_Conv_15
from vae.models_norm import Norm_3d_Conv_15
from PIL import Image
import torchvision.transforms as transforms


class Screen():
    def __init__(self, features, model_name, zdim, xydim, datasetsize, use_neg=False):
        self.features = features
        self.use_neg = use_neg
        bins = norm.ppf(np.linspace(0, 1, 65))
        bins = np.delete(bins, 0,0)
        self.bins = np.delete(bins, -1,0)
        self.index = [0,1]

        self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((datasetsize,datasetsize)),
                transforms.ToTensor()]
            )

        if self.features == "model":
            
            print("Loading model")
            self.stat_dictionary = {}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Res_3d_Conv_15(zdim, ResidualBlock, datasetsize, 0.5, 1)
            self.model.load_state_dict(torch.load(model_name,map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            print("Done with loading model")
        
        if self.features == "model_gaus":
            print("Loading model")
            self.stat_dictionary = {}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Norm_3d_Conv_15(zdim, ResidualBlock, datasetsize)
            self.model.load_state_dict(torch.load(model_name,map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            print("Done with loading model")

    def GetFeatures(self,env, new_node, prev_node=None):
        if self.features == "model":
            return self.GetFeaturesFromModel(env, new_node, prev_node)
        elif self.features == "model_gaus":
            return self.Quantization(env,new_node,prev_node)
        else:
            return self.Bprost(env, new_node, prev_node)

    def Bprost(self, env, new_node, prev_node):
        one_dim_screen = np.array(env.ale.getScreen(), dtype="int32").flatten()
        if prev_node is not None:
            prev_basic_features = prev_node.data['features']
            bprost_features = bprost_lib.bprost(one_dim_screen, prev_basic_features, False)
            bprost_features = np.asarray(bprost_features, dtype="int32")
            new_node.data["features"] = bprost_features
        else:
            bprost_features = bprost_lib.bprost(one_dim_screen, np.array([0], dtype="int32"), True)
            bprost_features = np.asarray(bprost_features, dtype="int32")
            new_node.data["features"] = bprost_features

    def UniqueFeatures(self):
        return len(self.stat_dictionary)

    def GetFeaturesFromModel(self, env, new_node, prev_node=None):
        with torch.no_grad():
            image = Image.fromarray((env.unwrapped._get_obs()).astype(np.uint8))
            transformed_image = self.transform(image)
            device_image = transformed_image.unsqueeze(0).to(self.device)
            tensor_features = self.model.get_features(device_image)
            features = tensor_features[0].cpu().numpy()
            features = np.where(features > 0.9,1,0)
            if self.use_neg:
                temp = 1-np.copy(features)
                features = np.append(temp,features)  
            
            indexFeatures = np.where(features)[0]
            new_node.data["features"] = indexFeatures

    def Quantization(self, env,new_node,prev_node=None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(env.unwrapped._get_obs()),'L')
            transformed_image = self.transform(image)
            device_image = transformed_image.unsqueeze(0).to(self.device)
            mu = self.model.get_features(device_image)
            mu = mu[0].cpu().numpy()
            # Quantization of the features
            inds = np.digitize(mu, self.bins)
            inds = np.array(inds, dtype=np.uint8)
            inds = np.expand_dims(inds,axis=1)
            binary_rep_8 = np.unpackbits(inds, axis=1,bitorder='big')
            #Make it 4 bits size
            binary_rep_4 = np.delete(binary_rep_8, self.index, 1).flatten()
            indexFeatures= np.where(binary_rep_4)[0]
            new_node.data["features"] = indexFeatures