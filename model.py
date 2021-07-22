import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, base_model):
        super(Regressor,self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=960, out_features=4)
        
    def forward(self,x):
        x = self.base(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x
    
