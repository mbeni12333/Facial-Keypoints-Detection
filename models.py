## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from torchvision.models import vgg19

class VGG_reg(nn.Module):
    """
        This model uses a pretrained vgg for feature extration, then passe it to a regressor that will actually learn the mapping, 
        The Feature detection tool, can eventually be finetuned to improve the loss
    """
    def __init__(self):
        
        super(VGG_reg, self).__init__()
        
        # get the feature_detector part
        self.features = vgg19(pretrained=True).features

        for param in self.features.parameters():
            param.requires_grad = False
            
        self.regressor = nn.Sequential(
            nn.Linear(25088, 2048),nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(2048, 1024),nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.6), 
            nn.Linear(1024, 136)
        )
    
    def forward(self, x):
        
        # detect features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        
        
        # Regressor
        x = self.regressor(x)
        
        return x
    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 5), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 5), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.4),
        )
        
#         self.features = vgg19(pretrained=True).features
#         for param in self.features.parameters():
#             param.requires_grad = False
            
#         self.conv1 = nn.Conv2d(1, 32, 4)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.conv3 = nn.Conv2d(64, 128, 2)
#         self.conv4 = nn.Conv2d(128, 256, 1)
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(30976, 2048),nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(2048, 1024),nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.6), 
            nn.Linear(1024, 136)
        )
#         self.fc1 = nn.Linear(, 1000)
#         self.fc2 = nn.Linear(1000, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        
        # feature extraction
        x = self.features(x)
        
        # Flatten
        x = x.view(x.shape[0],-1)
        
        
        #regression
        x = self.regressor(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
