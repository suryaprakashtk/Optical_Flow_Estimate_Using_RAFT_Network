import numpy as np
import torch
import torch.utils.data as data
from os.path import *
from glob import glob
import os.path as osp
import os

from Read_Write_Files import *
from torchvision.transforms import transforms

class BaseDataloader(data.Dataset):
    def __init__(self, path, mode = 'training'):
        self.image_path = []
        self.flow_path = []
        self.is_test = False
        self.transform = transforms.Resize((384,512))
        self.mode = mode
        self.path = path

    def __getitem__(self, ind):     

        flow_gt = read_gen(self.flow_path[ind])
        img1 = read_gen(self.image_path[ind][0])
        img2 = read_gen(self.image_path[ind][1])
        flow_gt = np.array(flow_gt).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # Convert image to grayscale
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]            

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow_gt = torch.from_numpy(flow_gt).permute(2, 0, 1).float()

        return self.transform(img1), self.transform(img2), self.transform(flow_gt)
    
    def __rmul__(self, num):
        self.image_path = num * self.image_path
        self.flow_path = num * self.flow_path
        return self

    def __len__(self):
        return len(self.image_path)
    
    def crowd(self):
        path_crowd = self.path + 'crowdflow/' + self.mode

        image = sorted(glob(osp.join(path_crowd, 'images', '*.png')))
        flow = sorted(glob(osp.join(path_crowd, 'flow', '*.flo')))
        
        for i in range(len(image)-1):
            self.image_path.append([image[i], image[i+1]])
            self.flow_path.append(flow[i])
        return

def get_train_dataloader(dataset_dir):
    dataset = BaseDataloader(dataset_dir, mode='training')
    dataset.crowd()

    train_loader = data.DataLoader(dataset, batch_size=1)
    return train_loader

def get_val_dataloader(dataset_dir):
    dataset = BaseDataloader(dataset_dir, mode='validation')
    dataset.crowd()

    val_loader = data.DataLoader(dataset, batch_size=1)
    return val_loader

def get_test_dataloader(dataset_dir):
    dataset = BaseDataloader(dataset_dir, mode='testing')
    dataset.crowd()

    test_loader = data.DataLoader(dataset, batch_size=1)
    return test_loader