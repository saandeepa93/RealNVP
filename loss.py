import torch
from torch import nn
import numpy as np

from sys import exit as e

class RealNVPLoss(nn.Module):
  def __init__(self, k=256):
    super(RealNVPLoss, self).__init__()
    self.k = k

  def forward(self, z, sldj):
    prior_z = -0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_z = prior_z.reshape(z.size(0), -1).sum(-1) -\
      np.log(self.k) * np.prod(z.size()[1:])
    ll = prior_z + sldj
    nll = -ll.mean()
    return nll