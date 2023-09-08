import PIL.Image
import random
import torch
import torch.utils.data
import numpy as np
from collections import defaultdict
import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import models

import torchvision.transforms as original_transforms
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
import multiprocessing as mp
from torch import nn
import torch.optim as optim
from tqdm import tqdm

import pycocotools



n_gpus = torch.cuda.device_count()
USING_CPU = not torch.cuda.is_available()

DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()  and n_gpus > 0) else "cpu")
kwargs = {'num_workers': mp.cpu_count() , 'pin_memory': True} if DEVICE.type=='cuda' else {'num_workers': mp.cpu_count()//2, 'prefetch_factor': 4}

print(f'Num of CPUs: {mp.cpu_count()}')
print(f'Device in use: {DEVICE}')
print(f'Found {n_gpus} GPU Device/s.')

TRAIN_IMG_DIR = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
TRAIN_ANN_FILE = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
USE_PRETRAINED = True

def load_dataset(transform):
    return dset.CocoDetection(root = TRAIN_IMG_DIR, 
                              annFile = TRAIN_ANN_FILE)

#Fucntions to modify the images (flip/mirror/resize)
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.hf = transforms.RandomHorizontalFlip(1)
        
    def __call__(self, img, bboxes):
        if torch.rand(1)[0] < self.p:            
            img = self.hf.forward(img)
            bboxes = self.hf.forward(bboxes)
        return img, bboxes
    
    
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.vf = transforms.RandomVerticalFlip(1)
        
    def __call__(self, img, bboxes):
        if torch.rand(1)[0] < self.p:                    
            img = self.vf.forward(img)
            bboxes = self.vf.forward(bboxes)
        return img, bboxes

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(self.size, antialias=True)
        
    def __call__(self, img, bboxes):
        img = self.resize.forward(img)
        bboxes = self.resize.forward(bboxes)
        return img, bboxes

def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    
    resize = Resize((300, 300))
    
    rhf = RandomHorizontalFlip()
    rvf = RandomVerticalFlip()
    image, target = sample
    
    image, bboxes = image,target["boxes"] 
    
    image, bboxes = resize(image, bboxes)
    image, bboxes = rhf(image, bboxes)
    image, bboxes = rvf(image, bboxes)
    
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
        
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, bboxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()

    