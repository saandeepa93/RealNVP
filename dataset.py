import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io

from utils import show

from sys import exit as e

class CustomDataset(Dataset):
  def __init__(self, root_folder, size):
    super(CustomDataset, self).__init__()
    self.root_folder = root_folder
    self.all_files = glob.glob(os.path.join(self.root_folder, '*.jpg'))
    self.transform = transforms.Compose(
      [transforms.ToPILImage(),
      transforms.Resize((size, size)),
      transforms.ToTensor(),
      transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    file_name = self.all_files[idx]
    if os.path.splitext(file_name)[-1] == ".jpg":
      img = io.imread(file_name)
      img = self.transform(img)
    return img


