import torch
import torch.nn as nn
import torch.nn.functional as F

#卷积神经网络模型
class FruitCNNmodel(nn.Module):
    def __init__(self):
        super(FruitCNNmodel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),  
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),        
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16 , 1024),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
