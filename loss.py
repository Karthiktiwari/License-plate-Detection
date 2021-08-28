import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self,):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_noobj = 0.5
        self.lambda_obj = 5

    def forward(self, outputs, targets):
        obj = targets[..., 0] == 1  
        noobj = targets[..., 0] == 0

        class_loss = self.bce((outputs[..., 0:2][obj]), targets[..., 0:2][obj].long())

        wh_loss = (self.mse((outputs[..., 5:6][obj]) ** 1/2, targets[..., 5:6][obj]) ** 1/2) +  
                  (self.mse((outputs[..., 6:][obj]) ** 1/2, targets[..., 6:][obj]) ** 1/2)

        xy_loss = (self.mse((outputs[..., 3:4][obj]), targets[..., 3:4][obj])) +  (self.mse((outputs[..., 4:5][obj]), targets[..., 4:5][obj]))
        # box_loss = self.mse((outputs[..., 3:][obj]), targets[..., 3:][obj])

        noclass_loss = self.bce((outputs[..., 0:2][noobj]), targets[..., 0:2][noobj].long())
        noclass_prob_loss = self.mse((outputs[..., 2:3][noobj]), targets[..., 2:3][noobj])

        total_loss = lambda_obj * xy_loss +
                     lambda_obj * wh_loss +
                     class_loss + 
                     lambda_noobj * noclass_loss +
                     noclass_prob_loss

        return total_loss