import os
import argparse
from torch import log
import matplotlib.pyplot as plt
import shutil

import torch 
from torch import optim, nn
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

from model import Resnet, Coupling, RealNVP, Flow
from dataset import CustomDataset
from utils import show, get_config, grad_clipping
from loss import RealNVPLoss

from sys import exit as e

def save_checkpoint(state, is_best, root, filename='./models/checkpoint.pth.tar'):
  torch.save(state, os.path.join(root, filename))
  if is_best:
    shutil.copyfile(os.path.join(root, filename), os.path.join(root, './models/model_best.pth.tar'))

def test(model_path, model):
  device = torch.device('cpu')
  model = Flow(in_channel, 64, n_block, n_flows, img_sz, num_scales)
  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint)
  z_sample = torch.randn((16, 3, 32, 32), dtype=torch.float32)
  x = model.reverse(z_sample)
  x = torch.sigmoid(x)
  show(x[0].permute(1, 2, 0).detach().cpu(), "test", 0)
  # print(x.size())



if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
  parser = argparse.ArgumentParser()
  parser.add_argument("--no_cuda", default=False, help="Specify if you want to use cuda")
  parser.add_argument("--root", default="./")
  opt = parser.parse_args()

  root = opt.root
  config_path = os.path.join(opt.root, "config.yaml")
  configs = get_config(config_path)
  batch_size = configs["params"]["batch_size"]
  n_block = configs["params"]["n_block"]
  n_flows = configs["params"]["n_flows"]
  num_scales = configs["params"]["num_scales"]
  in_channel = configs["params"]["in_channel"]
  lr = float(configs["params"]["lr"])
  data_root = configs["params"]["data_root"]
  img_sz = configs["params"]["img_sz"]
  epochs = configs["params"]["epochs"]
  test_model = configs["params"]["test"]
  model_path = configs["params"]["model_path"]

  use_cuda = (torch.cuda.is_available())
  device = "cuda" if use_cuda else "cpu"
  kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}
  
    

  # transform = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor()])
  # train_dataset = MNIST(root=opt.root, train=True, transform=transform, \
  #   download=True)
  # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
  # **kwargs)

  dataset = CustomDataset(data_root, img_sz)
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  model = Flow(in_channel, 64, n_block, n_flows, img_sz, num_scales)

  if test_model:
    test(os.path.join(root, model_path), model)
  
  else:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = RealNVPLoss()
    
    if device=="cuda" and torch.cuda.device_count()>1:
      model = DataParallel(model, device_ids=[0, 1, 2, 3])
      model = model.to(device)


    old_loss = 1e6
    best_loss = 0
    is_best = 0
    for epoch in range(epochs):
      for b, x in enumerate(train_loader, 0):
        model.train()
        optimizer.zero_grad()
        x = x.to(device)
        # show(x[0].detach().permute(1, 2, 0), os.path.join(root, "input/1.png"), 0)
        z, sldj = model(x)
        # x = model.reverse(z)
        # show(x[0].detach().permute(1, 2, 0), os.path.join(root, "input/1.png"), 0)
        loss = loss_fn(z, sldj)
        loss.backward()
        grad_clipping(optimizer)
        optimizer.step()
        if loss.item() < old_loss:
          is_best = 1
          save_checkpoint(model.state_dict(), is_best, root, f"./models/checkpoint.pth.tar")
          old_loss = loss
        print(f"loss at batch {b} epoch {epoch}: {loss.item()}")
        if b == 0: #len(train_loader) - 1:
          z_sample = torch.randn((16, 3, 32, 32), dtype=torch.float32).to(device)
          model.eval()
          if use_cuda:
            x = model.module.reverse(z_sample)
          else:
            x = model.reverse(z_sample)
          images = torch.sigmoid(x)
          images_concat = utils.make_grid(images, nrow=int(16 ** 0.5), padding=2, pad_value=255)
          utils.save_image(images_concat, os.path.join(root, f"./samples/{epoch}_{b}_{str((torch.round(loss * 10**3)/10**3).item())}.png"))
          # show(x.squeeze().detach().cpu(), os.path.join(root, f"./samples/{epoch}_{b}_{str((torch.round(loss * 10**3)/10**3).item())}.png"), 1)
          # show(x.squeeze().permute(1, 2, 0).detach().cpu(), os.path.join(root, f"./samples/{epoch}_{b}_{str((torch.round(loss * 10**3)/10**3).item())}.png"), 1)
    best_loss = old_loss
    print(f"Best loss at {best_loss}")

      