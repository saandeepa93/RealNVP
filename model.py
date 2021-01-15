from torch import nn, log
import torch 
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from math import pi, log

from utils import checkerboard_mask
from sys import exit as e


class Rescale(nn.Module):
  def __init__(self, num_channels):
    super(Rescale, self).__init__()
    self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

  def forward(self, x):
    x = x * self.weight
    return x

class WNConv2d(nn.Module):
  def __init__(self, in_channel, mid_channel, kernel, padding, bias):
    super(WNConv2d, self).__init__()
    self.conv = weight_norm(nn.Conv2d(in_channel, mid_channel, kernel_size=kernel, padding=padding, bias=bias))

  def forward(self, x):
    return self.conv(x)

class ResidualBlock(nn.Module):
  def __init__(self, in_channel, mid_channel):
    super(ResidualBlock, self).__init__()
    self.in_norm = nn.BatchNorm2d(in_channel)
    self.in_conv = WNConv2d(in_channel, mid_channel, kernel=3, padding=1, bias = False)

    self.out_norm = nn.BatchNorm2d(mid_channel)
    self.out_conv = WNConv2d(mid_channel, mid_channel, kernel=3, padding=1, bias=True)

  def forward(self, x):
    skip = x
    x = self.in_norm(x)
    x = F.relu(x)
    x = self.in_conv(x)
    x = self.out_norm(x)
    x = F.relu(x)
    x = self.out_conv(x)
    x = x + skip
    return x

  
class Resnet(nn.Module):
  def __init__(self, in_channel, mid_channel, n_block):
    super(Resnet, self).__init__()
    self.in_norm = nn.BatchNorm2d(in_channel)
    self.in_conv = WNConv2d(2 * in_channel, mid_channel, kernel=3, padding=1, bias=True)

    self.resnet = nn.ModuleList()
    for _ in range(n_block):
      self.resnet.append(ResidualBlock(mid_channel, mid_channel))
    
    self.skips = nn.ModuleList()
    for _ in range(n_block):
      self.skips.append(WNConv2d(mid_channel, mid_channel, kernel=1, padding=0, bias=True))

    self.out_norm = nn.BatchNorm2d(mid_channel)
    self.out_conv = WNConv2d(mid_channel, 2 * in_channel, kernel=1, padding=0, bias=True)
    
  
  def forward(self, x):
    x = self.in_norm(x)
    # To make the mean zero
    x = torch.cat([x, -x], 1)
    x = F.relu(x)
    x = self.in_conv(x)

    for resblock in self.resnet:
      x = resblock(x)
    
    x = self.out_norm(x)
    x = F.relu(x)
    x = self.out_conv(x)
    return x


class Coupling(nn.Module):
  def __init__(self, in_channel, mid_channel, n_block, n_flows, img_sz, parity, mask_type):
    super(Coupling, self).__init__()
    self.img_size = img_sz
    self.parity = parity
    self.mask_type = mask_type
    self.st_net = Resnet(in_channel, mid_channel, n_block)
    self.rescale = weight_norm(Rescale(in_channel))
    
  def forward(self, x, sldj):
    if self.mask_type == "cb":
      mask = checkerboard_mask(x.size(2), x.size(3))
      mask = mask.to(x.device)
      if self.parity:
        mask = 1 - mask
      x_a = mask * x
      st = self.st_net(x_a)
      s, t = st.chunk(2, 1)
      s = self.rescale(torch.tanh(s))
      s = s * (1 - mask)
      t = t * (1 - mask)
      exp_s = s.exp()
      x = (x + t) * exp_s
      sldj += s.contiguous().view(s.size(0), -1).sum(-1)
      return x, sldj
    elif self.mask_type == "cw":
      if self.parity:
        x_b, x_a = x.chunk(2, dim=1)
      else:
        x_a, x_b = x.chunk(2, dim=1)
      st = self.st_net(x_b)
      s, t = st.chunk(2, dim=1)
      s = self.rescale(torch.tanh(s))
      exp_s = s.exp()
      x_a = (x_a + t) * exp_s
      sldj += s.contiguous().view(s.size(0), -1).sum(-1)
      if self.parity:
        x = torch.cat([x_b, x_a], dim = 1)
      else:
        x = torch.cat([x_a, x_b], dim = 1)
      return x, sldj



  def reverse(self, z):
    if self.mask_type == "cb":
      mask = checkerboard_mask(z.size(2), z.size(3))
      mask = mask.to(z.device)
      if self.parity:
        mask = 1 - mask
      z_a = mask * z
      st = self.st_net(z_a)
      s, t = st.chunk(2, 1)
      s = self.rescale(torch.tanh(s))
      s = s * (1 -  mask)
      t = t * (1 - mask)
      inv_exp_s = s.mul(-1).exp()
      z = z * inv_exp_s - t
      return z
    elif self.mask_type == "cw":
      if self.parity:
        z_b, z_a = z.chunk(2, dim=1)
      else:
        z_a, z_b = z.chunk(2, dim=1)
      st = self.st_net(z_b)
      s, t = st.chunk(2, dim=1)
      s = self.rescale(torch.tanh(s))
      inv_exp_s = s.mul(-1).exp()
      z_a = z_a * inv_exp_s - t
      if self.parity:
        z = torch.cat([z_b, z_a], dim = 1)
      else:
        z = torch.cat([z_a, z_b], dim = 1)
      return z


class RealNVP(nn.Module):
  def __init__(self, scale_idx, in_channel, mid_channel, n_block, n_flows, img_sz, num_scales):
    super(RealNVP, self).__init__()
    self.is_last_block = (scale_idx == num_scales - 1)

    self.in_couplings = nn.ModuleList([Coupling(in_channel, mid_channel, n_block, n_flows, img_sz, False, "cb"),\
        Coupling(in_channel, mid_channel, n_block, n_flows, img_sz, True, "cb"), \
        Coupling(in_channel, mid_channel, n_block, n_flows, img_sz, False, "cb")
      ])
    
    if self.is_last_block:
      self.in_couplings.append(Coupling(in_channel, mid_channel, n_block, n_flows, img_sz, True, "cb"))
    
    else:
      self.out_couplings = nn.ModuleList([Coupling(2 * in_channel, mid_channel, n_block, n_flows, img_sz, False, "cw"),\
          Coupling(2 * in_channel, mid_channel, n_block, n_flows, img_sz, True, "cw"), \
          Coupling(2 * in_channel, mid_channel, n_block, n_flows, img_sz, False, "cw")
        ])
      self.next_block = RealNVP(scale_idx+1, 2*in_channel, 64, n_block, n_flows, img_sz, num_scales)
    

  def squeeze_2x2(self, x, reverse = False, alt_order = False):
    b, c, h, w, = x.size()
    if alt_order:
      if reverse:
        c = c//4
      
      squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
      squeeze_weights = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
      for c_idx in range(c):
        slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
        slice_1 = slice(c_idx, c_idx + 1)
        squeeze_weights[slice_0, slice_1, :, :] = squeeze_matrix
      shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                      + [c_idx * 4 + 1 for c_idx in range(c)]
                                      + [c_idx * 4 + 2 for c_idx in range(c)]
                                      + [c_idx * 4 + 3 for c_idx in range(c)])
      squeeze_weights = squeeze_weights[shuffle_channels, :, :, :]

      '''  
      squeeze_matrix = torch.tensor([[[[1, 0], [0, 0]]], \
                                    [[[0, 1], [0 ,0]]], \
                                    [[[0, 0], [1, 0]]], \
                                    [[[0, 0], [0, 1]]]], dtype = x.dtype, device = x.device)
      # squeeze_matrix = squeeze_matrix.squeeze()
      squeeze_weights = torch.zeros(c * 4, c, 2, 2, dtype=x.dtype, device=x.device)
      print(squeeze_weights.size(), squeeze_matrix.size())
      for ch in range(c):
        indeces = list(range(ch, c*4, 3))
        squeeze_weights[indeces, ch, :, :] = squeeze_matrix
    '''
      if reverse:
        x = F.conv_transpose2d(x, squeeze_weights, stride = 2)
      else:
        x = F.conv2d(x, squeeze_weights, stride=2)
    else:
      x = x.permute(0, 2, 3, 1)
      if reverse:
        x = x.view(b, h, w, c//4, 2, 2)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.contiguous().view(b, h*2, w*2, c//4)
      else:
        x = x.view(b, h//2, 2, w//2, 2, c)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous().view(b, h//2, w//2, c*4)
      x = x.permute(0, 3, 1, 2)
    return x

  def forward(self, x, sldj):
    for coupling in self.in_couplings:
      x, sldj = coupling(x, sldj)
    
    if not self.is_last_block:
      # print("1. ", x.size())
      x = self.squeeze_2x2(x, False)
      # print("2: ", x.size())
      for coupling in self.out_couplings:
        x, sldj = coupling(x, sldj)
      # print("3: ", x.size())
      x = self.squeeze_2x2(x, True)
      # print("4: ", x.size())
      x = self.squeeze_2x2(x, False, True)
      # print("5: ", x.size())
      x, x_split = x.chunk(2, dim = 1)
      # print("6: ", x.size())
      x, sldj = self.next_block(x, sldj)
      # print("7: ", x.size())
      x = torch.cat([x, x_split], dim = 1)
      # print("8: ", x.size())
      x = self.squeeze_2x2(x, True, True)
      # print("9: ", x.size())
    return x, sldj

  def reverse(self, z):
    if not self.is_last_block:
      z = self.squeeze_2x2(z, False, True)
      z, z_split = z.chunk(2, dim = 1)
      z = self.next_block.reverse(z)
      z = torch.cat([z, z_split], 1)
      z = self.squeeze_2x2(z, True, True)
      z = self.squeeze_2x2(z)
      for coupling in self.out_couplings[::-1]:
        z = coupling.reverse(z)
      z = self.squeeze_2x2(z, True)
    for coupling in self.in_couplings[::-1]:
      z = coupling.reverse(z)
    return z
  
class Flow(nn.Module):
  def __init__(self, in_channel, mid_channel, n_block, n_flows, img_sz, num_scales):
    super(Flow, self).__init__()
    self.flows = RealNVP(0, in_channel, 64, n_block, n_flows, img_sz, num_scales)

  def preprocess(self, x):
    self.data_constraint = torch.tensor([0.9], dtype=float, device = x.device)
    y = (x * 255. + torch.rand_like(x)) / 256.
    y = (2 * y - 1) * self.data_constraint
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
        - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
    sldj = ldj.contiguous().view(ldj.size(0), -1).sum(-1)
    return y.type(torch.float32), sldj
    
  def forward(self, x):
    x, sldj = self.preprocess(x)
    x, sldj = self.flows(x, sldj)
    return x, sldj
  
  def reverse(self, z):
    z = self.flows.reverse(z)
    return z

  






    
      



      