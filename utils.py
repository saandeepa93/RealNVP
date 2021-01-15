import yaml
import matplotlib.pyplot as plt

import torch
from torch.nn.utils import clip_grad_norm


def get_config(config_path):
  with open(config_path) as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs

def show(img, name, flg):
  plt.imshow(img)
  if flg:
    plt.savefig(name)
  else:
    plt.show()

def checkerboard_mask(h, w):
  mask = []
  for i in range(h * w):
    mask.append(i%2)
  mask = torch.tensor(mask)
  return mask.view(h, w)

def grad_clipping(optimizer, max_norm = 100, norm_type=2):
  for group in optimizer.param_groups:
    clip_grad_norm(group["params"], max_norm, norm_type)