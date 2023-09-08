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

TRAIN_IMG_DIR = './images/train2017'
TRAIN_ANN_FILE = './images/instances_train2017.json'
USE_PRETRAINED = True
SAVED_MODEL_PATH = "./model/ssd300_vgg16_checkpoint"

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

transform = transforms.Compose(
    [
        transforms.RandomPhotometricDistort(),        
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)

coco_train = load_dataset(transform=transform)
coco_train = dset.wrap_dataset_for_transforms_v2(coco_train)

class NewCocoDataset(Dataset):    
    def __init__(self, coco_dataset, image_size=(312, 312)):
        """
        Arguments:
            coco_dataset (dataset): The coco dataset containing all the expected transforms.
            image_size (tuple): Target image size. Default is (512, 512)
        """
        
        self.coco_dataset = coco_dataset
        self.resize = Resize(image_size)
        self.rhf = RandomHorizontalFlip()
        self.rvf = RandomVerticalFlip()   
        self.transformer = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        
    def __len__(self):
        return len(self.coco_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        new_target = {}
        
        image, target = self.coco_dataset[idx]
        
        if 'boxes' not in target:    
            new_idx = idx-1
            _img, _t = self.coco_dataset[new_idx]
            while 'boxes' not in _t :
                new_idx -= 1
                _img, _t = self.coco_dataset[new_idx]
                
            image, target = self.coco_dataset[new_idx]
        
        
        image, bboxes = image, target["boxes"] 
            
        image, bboxes = self.resize(image, bboxes)
        image, bboxes = self.rhf(image, bboxes)
        image, bboxes = self.rvf(image, bboxes)
        
        image = self.transformer(image)
        
        new_boxes = []
        for box in bboxes:
            if box[0] < box[2] and box[1] < box[3]:
                new_boxes.append(box)
        
        new_target["boxes"] = torch.stack(new_boxes)
        new_target["labels"] = target["labels"]
    
        return (image, new_target)

new_coco_train = NewCocoDataset(coco_train)

data_loader = torch.utils.data.DataLoader(
    new_coco_train,
    batch_size=20,
    shuffle=True,
)

import pycocotools.coco

coco_anns = pycocotools.coco.COCO(TRAIN_ANN_FILE)
catIDs = coco_anns.getCatIds()
cats = coco_anns.loadCats(catIDs)

name_idx = {}

for sub_dict in cats:
    name_idx[sub_dict["id"]] = sub_dict["name"]
    
del coco_anns, catIDs, cats

data = next(iter(data_loader))
if USING_CPU:
    x = torch.stack(data[0])
else:
    x = data[0]
print(x.shape)

plt.imshow(data[0][0].permute(1, 2, 0).numpy())

data[1][0]['boxes']
data[0][0].shape, data[0][1].shape

base_model = models.get_model("ssd300_vgg16", weights=None, weights_backbone=None).train()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

base_model.apply(weights_init)
if (DEVICE.type == 'cuda') and (n_gpus > 1):
    base_model = nn.DataParallel(base_model, list(range(n_gpus)))

base_model.to(DEVICE)

total_params = sum(p.numel() for p in base_model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in base_model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

learning_rate = 1e-4

optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
if USE_PRETRAINED:
    new_LR = 1e-5 # change this value to set a new Learning Rate for the version of notebook
    
    if USING_CPU:
        checkpoint = torch.load(SAVED_MODEL_PATH, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(SAVED_MODEL_PATH)
        
    base_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for g in optimizer.param_groups:
        g['lr'] = new_LR

EPOCHS = 2

import gc

for epoch in range(EPOCHS):
    running_classifier_loss = 0.0
    running_bbox_loss = 0.0
    running_loss = 0.0
    
    counter = 0
    base_model.train()
    
    for data_point in tqdm(data_loader):
        _i, _t = data_point[0], data_point[1]
        
        if USING_CPU:
            _i = torch.stack(_i)

        _i = _i.to(DEVICE)
        _t = [{k: v.to(DEVICE) for k, v in __t.items()} for __t in _t]
        optimizer.zero_grad()
        loss_dict = base_model(_i, _t)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        del loss_dict, losses
        counter += 1
        if counter % 500 == 499:
            last_classifier_loss = running_classifier_loss / 500 # loss per batch
            last_bbox_loss = running_bbox_loss / 500 # loss per batch
            last_loss = running_loss / 500 # loss per batch
            print(f'Epoch {epoch}, Batch {counter + 1}, Running Loss: {last_loss}')
            running_classifier_loss = 0.0
            running_bbox_loss = 0.0
            running_loss = 0.0
            
        gc.collect()

VAL_IMG_DIR = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
VAL_ANN_FILE = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json'


def load_val_dataset(transform):
    return dset.CocoDetection(root = VAL_IMG_DIR, 
                              annFile = VAL_ANN_FILE)

val_transform = transforms.Compose(
    [
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)
coco_val = load_val_dataset(transform=val_transform)
coco_val = dset.wrap_dataset_for_transforms_v2(coco_val)

new_coco_val = NewCocoDataset(coco_val)
val_data_loader = torch.utils.data.DataLoader(
    new_coco_val,
    batch_size=50 if not USING_CPU else 8,
    shuffle=True,
#     collate_fn=lambda batch: tuple(zip(*batch)),
    collate_fn=collate_wrapper,
     **kwargs
)

img_dtype_converter = transforms.ConvertImageDtype(torch.uint8)
data = next(iter(val_data_loader))

_i = data[0]

threshold = 0.5
idx = 3

if USING_CPU:
    _i = torch.stack(_i)

_i = _i.to(DEVICE)
base_model.eval()
p_t = base_model(_i)

confidence_length = len(np.argwhere(p_t[idx]['scores'] > threshold)[0])

p_boxes = p_t[idx]['boxes'][: confidence_length]
p_labels = [name_idx[i] for i in p_t[idx]['labels'][: confidence_length].tolist()]
i_img = img_dtype_converter(_i[idx])

annotated_image = draw_bounding_boxes(i_img, p_boxes, p_labels, colors="yellow", width=3)
fig, ax = plt.subplots()
ax.imshow(annotated_image.permute(1, 2, 0).numpy())
ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
fig.tight_layout()


fig.show()

PATH = './model/ssd300_vgg16_checkpoint'

torch.save({
            'epoch': EPOCHS,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

checkpoint = torch.load(PATH)
base_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])