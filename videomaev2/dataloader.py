import os
import cv2
import torch
import random
import torchvision
import numpy as np
import pandas as pd
from os import path
from torch import nn
from torchvision import transforms
from decord import cpu, VideoReader
from torchvision.transforms import v2
        

class PreflightWindowDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv: str, mode: str, augmentate_p=0.25):
        self.sample_index = pd.read_csv(path_to_csv, index_col=0)
        self.resize_transform = transforms.Resize((224, 224), antialias=True)

        self.augmentate_p = augmentate_p

        self.augmentations = transforms.Compose([
                transforms.RandomAutocontrast(augmentate_p),
                transforms.RandomEqualize(augmentate_p),
                transforms.RandomPosterize(6, augmentate_p),
                transforms.v2.RandomPhotometricDistort(
                    brightness=[0.85, 1.2],
                    contrast=[0.8, 1.2],
                    saturation=[0.8, 1.2],
                    hue=[-0.02, 0.02],
                    p=augmentate_p
                )
        ])

        self.to_augmentate = False
        
        if mode == 'train':
            self.to_augmentate = True

    def __getitem__(self, index):
        file_path, action_start_frame, action_length, action_label = self.sample_index.iloc[index]
        
        vr = VideoReader(file_path, ctx=cpu(0))
        
        indices = random.sample(range(action_start_frame, action_start_frame + action_length), 16)
        indices.sort()

        frames = vr.get_batch(indices).asnumpy()        
        frames = torch.tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        
        if self.to_augmentate:
            frames = self.augmentations(frames)
            
        frames = self.resize_transform(frames).type(torch.float32)
        frames = frames / 255.
        
        if action_label == 11:
            return frames, torch.as_tensor([0]* 11)
        return frames, torch.nn.functional.one_hot(torch.as_tensor(action_label), num_classes=11)
    
    def __len__(self):
        return len(self.sample_index)

def create_dataloader(path_to_csv: str, batch_size: int, mode: str, drop_last: bool):
    dataset = PreflightWindowDataset(path_to_csv, mode=mode)    
    
    sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True)
    
    