import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)

    def forward(self, input, target):
        return self.mmd_loss(input, target)