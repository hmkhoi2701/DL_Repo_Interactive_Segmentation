import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset

from utils import random_click
from itertools import combinations
import nibabel as nib



class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        # if mode == 'Training':
        #     self.subfolders += [f.path for f in os.scandir(os.path.join(data_path, 'Validation' + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup, selected_rater_cup, not_selected_rater_cup, selected_rater_mask_cup, selected_rater_mask_cup_ori = random_click(multi_rater_cup, self.mask_size)
            


        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'selected_rater': selected_rater_cup, 
            'not_selected_rater': not_selected_rater_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }
        
class MBHSeg_Binary(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train',prompt = 'click', plane = False):
        self.data_path = data_path
        self.max_raters = 4
        
        gt_folder = os.path.join(data_path, mode, 'gt')
        img_folder = os.path.join(data_path, mode, 'imgs')
        
        gt_subfolders = []
        img_subfolders = []
        for f in os.scandir(gt_folder):
            if not f.is_dir():
                continue
            gt_subfolders.append(os.path.join(gt_folder, f.name))
            img_subfolders.append(os.path.join(img_folder, f.name))
            
        self.img_paths = []
        self.gt_paths = []
        for folder in img_subfolders:
            for f in os.listdir(folder):
                prefix = f[:-4]
                self.img_paths.append(os.path.join(folder, f))
                self.gt_paths.append([os.path.join(gt_folder, folder.split('/')[-1], prefix + '_r'+str(i)+'.npy') for i in range(1, 5) 
                         if os.path.exists(os.path.join(gt_folder, folder.split('/')[-1], prefix + '_r'+str(i)+'.npy'))])
        
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        
        """Get the images"""
        img_path = self.img_paths[index]
        gt_paths = self.gt_paths[index]
        
        img = Image.open(img_path).convert('RGB')
        multi_rater = [Image.fromarray(np.load(path)).convert('L') for path in gt_paths]
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup, selected_rater_cup, not_selected_rater_cup, selected_rater_mask_cup, selected_rater_mask_cup_ori = random_click(multi_rater_cup, self.mask_size)    
        
        def _torch_pad_first_dim(x, target_n):
            n = x.shape[0]
            if n == target_n:
                return x
            pad = torch.zeros((target_n - n, *x.shape[1:]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
            return x
        
        def _arr_pad_first_dim(x, target_n, val = 1.):
            # x: 1xn array
            n = x.shape[0]
            if n == target_n:
                return x
            pad = np.ones((target_n - n), dtype=x.dtype) * val
            x = np.concatenate([x, pad], axis=0)
            return x
        
        multi_rater_cup = _torch_pad_first_dim(multi_rater_cup, self.max_raters)
        
        selected_rater_cup = _arr_pad_first_dim(selected_rater_cup, self.max_raters, val = 0.)
        not_selected_rater_cup = _arr_pad_first_dim(not_selected_rater_cup, self.max_raters)
        
        image_meta_dict = {'filename_or_obj':img_path.split('/')[-1]}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'selected_rater': selected_rater_cup, 
            'not_selected_rater': not_selected_rater_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }
    
class LIDC(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train', prompt = 'click', plane = False):
        self.data_path = data_path
        
        gt_folder = os.path.join(data_path, mode, 'gt')
        img_folder = os.path.join(data_path, mode, 'images')
        
        non_empty_indices = np.load( f'assets/lidc_{mode}_non_empty_indices.npy')
        
        gt_subfolders = []
        img_subfolders = []
        for f in os.scandir(gt_folder):
            if not f.is_dir():
                continue
            gt_subfolders.append(os.path.join(gt_folder, f.name))
            img_subfolders.append(os.path.join(img_folder, f.name))
            
        self.img_paths = []
        self.gt_paths = []
        for folder in img_subfolders:
            for f in os.listdir(folder):
                prefix = f[:-4]
                self.img_paths.append(os.path.join(folder, f))
                self.gt_paths.append([os.path.join(gt_folder, folder.split('/')[-1], prefix + '_l'+str(i)+'.png') for i in range(0, 3)])
                
        self.img_paths = [self.img_paths[i] for i in non_empty_indices]
        self.gt_paths = [self.gt_paths[i] for i in non_empty_indices]
                
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        """Get the images"""
        img_path = self.img_paths[index]
        gt_paths = self.gt_paths[index]
        
        img = Image.open(img_path).convert('RGB')
        multi_rater = [Image.open(path).convert('L') for path in gt_paths]
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup, selected_rater_cup, not_selected_rater_cup, selected_rater_mask_cup, selected_rater_mask_cup_ori = random_click(multi_rater_cup, self.mask_size)
            


        image_meta_dict = {'filename_or_obj':img_path.split('/')[-1]}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'selected_rater': selected_rater_cup, 
            'not_selected_rater': not_selected_rater_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }