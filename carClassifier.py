import torch
from torch import nn
import torchvision
from torchvision import models

class CarTypeClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CarTypeClassifier, self).__init__()
        #Load ImageNet weights
        self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        #Change classification head to fully connected layer with 10 output nodes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        #Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Unfreeze the last two blocks in features for fine tuning
        for i in range(7, 9):
            for param in self.model.features[i].parameters():
                param.requires_grad = True


    def forward(self, x):
        return self.model(x)
