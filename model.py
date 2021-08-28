import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, base_model, S, C, img_size = 768):
        super(Regressor,self).__init__()

        self.S = S
        self.C = C
        self.img_size = img_size
        self.base = nn.Sequential(*list(base_model.children())[:-1])

        self.in_features = self.base(torch.rand(1, 3, self.img_size, self.img_size)).shape[1] #Output shape of fc of base model

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = self.in_features, out_features = 1024)
        self.fc2 = nn.Linear(in_features= 1024, out_features = (self.S*self.S*(self.C+5)))
        
    def forward(self,x):
        x = self.base(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.reshape(-1, self.S, self.S, self.C+5)

