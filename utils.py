""" helper function

author Jiayuan
"""

import collections
import logging
import math
import os
import pathlib
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import dateutil.tz
import matplotlib.pyplot as plt
import numpy
import numpy as np
import PIL
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from monai.config import print_config
from monai.data import (CacheDataset, ThreadDataLoader, decollate_batch,
                        load_decathlon_datalist, set_track_meta)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (AsDiscrete, Compose, CropForegroundd,
                              EnsureTyped, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
                              RandShiftIntensityd, ScaleIntensityRanged,
                              Spacingd)
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch import autograd
from torch.autograd import Function, Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from tqdm import tqdm

import cfg
import pandas as pd


args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)


def get_network(args, use_gpu=True, gpu_device = 0, distribution = True):
    from models.sam import SamPredictor, sam_model_registry
    from models.sam.utils.transforms import ResizeLongestSide
    net = sam_model_registry['default'](args,checkpoint=args.sam_ckpt).to(device)

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    os.makedirs(log_dir, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{time_str}_{phase}.log'
    final_log_file = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(message)s',
        handlers=[
            logging.FileHandler(final_log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger()  # root logger
    logger.info('Logger is ready.')
    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    # sample_path = os.path.join(prefix, 'Samples')
    # os.makedirs(sample_path)
    # path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2: # for REFUGE multi mask output
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    elif c > 2: # for multi-class segmentation > 2 classes
        preds = []
        gts = []
        for i in range(0, c):
            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            preds.append(pred)
            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            gts.append(gt)
        tup = [imgs[:row_num,:,:,:]] + preds + gts
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                
                gt_masks[i,0,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.5
                gt_masks[i,1,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.1
                gt_masks[i,2,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.4
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)


def random_click(multi_rater):

    multi_rater_mean = np.mean(np.array(multi_rater.squeeze(1)), axis=0)

    # with prob = 0.8 to choose divergent area, so subset of multi-rater, ow all agreement
    list_set = list(set(multi_rater_mean.flatten()))
    point_label = random.choice(list_set)
    if np.random.choice([True, False], 1, p=[0.8,0.2])[0]:
        divergent_values = [v for v in list_set if v != 0 and v != 1]
        if len(divergent_values) > 0:
            point_label = random.choice(divergent_values)
        
    # randomly select a position among these indices
    indices = np.argwhere(multi_rater_mean == point_label)
    pt =  indices[np.random.randint(len(indices))]
    
    # decide the label of the selected position, higher prob to be more rater agreement
    if (point_label != 0) and (point_label != 1):
        if np.random.choice([True, False], 1, p=[0.8,0.2])[0]:
            point_label = round(point_label)
        else:
            point_label = 1 - round(point_label)

    selected_rater_index = torch.nonzero(multi_rater[:,0,pt[0],pt[1]] == point_label).squeeze()
    not_selected_rater_index = torch.nonzero(multi_rater[:,0,pt[0],pt[1]] != point_label).squeeze()
    if selected_rater_index.ndim == 0:
        selected_rater_index = torch.tensor([selected_rater_index])
    if not_selected_rater_index.ndim == 0:
        not_selected_rater_index = torch.tensor([not_selected_rater_index])

    # ground truth: mean of selected raters' masks
    selected_rater_mask = multi_rater[selected_rater_index,:,:,:]
    selected_rater_mask = selected_rater_mask.mean(dim=0) # torch.Size([1, mask_size, mask_size])
    selected_rater_mask = (selected_rater_mask >= 0.5).float() # torch.Size([1, mask_size, mask_size])

    # propose point
    point_label, pt = agree_click(np.mean(np.array(multi_rater.squeeze(1)), axis=0), label = 1)

    # record selected and not selected raters' indices
    selected_rater_index = selected_rater_index.tolist()
    not_selected_rater_index = not_selected_rater_index.tolist()
    selected_rater = np.zeros(multi_rater.size(0))
    not_selected_rater = np.zeros(multi_rater.size(0))
    selected_rater[selected_rater_index] = 1
    not_selected_rater[not_selected_rater_index] = 1

    return point_label, pt, selected_rater, not_selected_rater, selected_rater_mask


def agree_click(mask, label = 1):
    # max agreement position
    indices = np.argwhere(mask == label) 
    if len(indices) == 0:
        label = 1 - label
        indices = np.argwhere(mask == label) 
    return label, indices[np.random.randint(len(indices))]

def random_click_multiclass(multi_rater: torch.Tensor):
    """
    multi_rater: (R, C, H, W), one-hot; 0=BG, 1..C-1=FG
    Return:
      point_label (int),
      pt (np.array [y,x]),
      selected_rater (np[R] 0/1),
      not_selected_rater (np[R] 0/1),
      selected_rater_mask (torch.FloatTensor [C,H,W], one-hot)
    """
    assert multi_rater.ndim == 4, "Expect (R, C, H, W)"
    R, C, H, W = multi_rater.shape
    device = multi_rater.device

    present_any = (multi_rater > 0.5).any(dim=0)          # (C,H,W) bool
    present_any_np = present_any.detach().cpu().numpy()
    
    diver_mask = (present_any.sum(dim=0) >= 2)
    diver_mask_np = diver_mask.detach().cpu().numpy()
    fg_classes = list(range(1, C))
    fg_in_div = [c for c in fg_classes if np.any(present_any_np[c] & diver_mask_np)]

    if len(fg_in_div)>0 and random.random() < 0.8:
        point_label = int(random.choice(fg_in_div))
    else:
        fg_present = [c for c in fg_classes if present_any[c].any().item()]
        point_label = int(random.choice(fg_present)) if fg_present else 0
        
    if point_label != 0:
        cand = np.argwhere(present_any_np[point_label] & diver_mask_np)
        if len(cand) == 0:
            cand = np.argwhere(present_any_np[point_label])
    else:
        cand = np.argwhere(diver_mask_np)
        if len(cand) == 0:
            cand = np.argwhere(present_any_np[0])
        if len(cand) == 0:
            cand = np.array([[np.random.randint(H), np.random.randint(W)]], dtype=np.int64)

    pt = cand[np.random.randint(len(cand))]
    y, x = int(pt[0]), int(pt[1])

    counts_at_pt = (multi_rater[:, :, y, x] > 0.5).sum(dim=0)   # (C,)
    present_at_pt = torch.nonzero(counts_at_pt > 0, as_tuple=False).squeeze(1).tolist()

    if present_at_pt:
        fg_present_at_pt = [c for c in present_at_pt if c != 0]
        if fg_present_at_pt:
            fg_counts = counts_at_pt[fg_present_at_pt]                  # (#fg,)
            max_fg = int(fg_counts.max().item())
            tie_idxs = (fg_counts == max_fg).nonzero(as_tuple=False).squeeze(1).tolist()
            maj_fg = int(fg_present_at_pt[random.choice(tie_idxs)])     # break tie trong FG
            if random.random() < 0.8:
                point_label = maj_fg
            else:
                fg_others = [c for c in fg_present_at_pt if c != maj_fg]
                point_label = int(random.choice(fg_others)) if fg_others else maj_fg
        else:
            point_label = 0
    else:
        point_label = 0


    sel_idx = torch.nonzero(multi_rater[:, point_label, y, x] > 0.5, as_tuple=False).squeeze()
    if sel_idx.ndim == 0 and sel_idx.numel() > 0:
        sel_idx = torch.tensor([sel_idx.item()], device=device)
    if sel_idx.numel() == 0:

        sel_idx = torch.arange(R, device=device)

    not_sel_idx = torch.tensor([i for i in range(R) if i not in set(sel_idx.tolist())],
                               device=device, dtype=torch.long)

    sel_mask_soft = multi_rater[sel_idx].float().mean(dim=0)     # (C,H,W)
    label_map = sel_mask_soft.argmax(dim=0)                      # (H,W)
    selected_rater_mask = F.one_hot(label_map, num_classes=C).permute(2,0,1).float().to(device)

    # Vectors 0/1
    selected_rater = np.zeros(R, dtype=np.float32)
    not_selected_rater = np.zeros(R, dtype=np.float32)
    if sel_idx.numel() > 0:
        selected_rater[sel_idx.detach().cpu().numpy()] = 1.0
    if not_sel_idx.numel() > 0:
        not_selected_rater[not_sel_idx.detach().cpu().numpy()] = 1.0

    return int(point_label), pt, selected_rater, not_selected_rater, selected_rater_mask

def random_box(multi_rater):
    max_value = torch.max(multi_rater[:,0,:,:], dim=0)[0]
    max_value_position = torch.nonzero(max_value)

    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]


    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))


    x_min = random.choice(np.arange(x_min-10,x_min+11))
    x_max = random.choice(np.arange(x_max-10,x_max+11))
    y_min = random.choice(np.arange(y_min-10,y_min+11))
    y_max = random.choice(np.arange(y_max-10,y_max+11))

    return x_min, x_max, y_min, y_max
