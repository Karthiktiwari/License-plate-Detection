import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, base_model):
        super(Regressor,self).__init__()

        self.base = nn.Sequential(*list(base_model.children())[:-1])

        self.in_features = self.base(torch.rand(1, 3, 512, 512)).shape[1] #Output shape of fc of base model

        self.flatten = nn.Flatten()
        self.vehicle_box_regressor = nn.Linear(in_features=self.in_features, out_features=4)
        self.plate_box_regressor = nn.Linear(in_features = self.in_features, out_features=4)
        
    def forward(self,x):
        x = self.base(x)
        x = self.flatten(x)
        vehicle_box = F.relu(self.vehicle_box_regressor(x))
        plate_box = F.relu(self.plate_box_regressor(x)) 
        return vehicle_box, plate_box

