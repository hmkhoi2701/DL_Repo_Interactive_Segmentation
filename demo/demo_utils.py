from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

def load_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)

def load_bin_mask(path, target_hw):
    H, W = target_hw
    m = Image.open(path).convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.NEAREST)
    arr = np.asarray(m, dtype=np.uint8)
    return (arr > 127).astype(np.float32)

def valid_raters(raters):
    return (raters.reshape(raters.shape[0], -1).sum(axis=1) > 0)

def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float=1e-7) -> Tuple[float,float]:
    pred = (pred > 0).astype(np.float32)
    gt   = (gt > 0).astype(np.float32)
    inter = float((pred*gt).sum())
    p_sum = float(pred.sum()); g_sum = float(gt.sum())
    union = p_sum + g_sum - inter
    if p_sum == 0 and g_sum == 0:  # empty-empty
        return 1.0, 1.0
    dice = (2.0*inter + eps) / (p_sum + g_sum + eps)
    iou  = (inter + eps) / (union + eps) if union > 0 else 1.0
    return dice, iou

def best_rater(pred: np.ndarray, raters: np.ndarray, valid: np.ndarray) -> Tuple[int, float]:
    best_i, best_d = -1, -1.0
    for i in range(raters.shape[0]):
        if not bool(valid[i]): 
            continue
        d,_ = dice_iou(pred, raters[i])
        if d > best_d:
            best_d = d; best_i = i
    return best_i, best_d

def neutral_binary(raters: np.ndarray, valid: np.ndarray) -> np.ndarray:
    w = valid.astype(np.float32).reshape(-1,1,1)
    denom = max(float(w.sum()), 1e-6)
    soft = (raters * w).sum(axis=0) / denom
    return (soft > 0.5).astype(np.float32)

def suggest_next_click(pred: np.ndarray, reference: np.ndarray) -> Optional[Tuple[int,int]]:
    diff = np.abs((pred>0).astype(np.uint8) - (reference>0).astype(np.uint8))
    if diff.sum() == 0:
        return None
    flat = int(np.argmax(diff))
    H,W = diff.shape
    return (flat//W, flat%W)

def overlay(image_rgb: np.ndarray,
            mask_bin: np.ndarray,
            clicks: List[Tuple[int,int,int]],
            sugg: Optional[Tuple[int,int]]=None) -> np.ndarray:
    img = image_rgb.copy().astype(np.float32)
    H,W,_ = img.shape
    assert mask_bin.shape == (H,W)
    alpha = 0.45
    mask3 = np.repeat(mask_bin[...,None], 3, axis=2).astype(np.float32)
    green = np.zeros_like(img); green[...,1] = 255
    img = img*(1 - alpha*mask3) + green*(alpha*mask3)
    out = img.astype(np.uint8)
    # draw points
    pil = Image.fromarray(out)
    r = 1
    draw = ImageDraw.Draw(pil)
    for (y,x,lbl) in clicks:
        color = (0,255,0) if int(lbl)==1 else (255,0,0)
        draw.ellipse((x-r,y-r,x+r,y+r), fill=color, outline=color)
    if sugg is not None:
        y,x = sugg
        draw.ellipse((x-r,y-r,x+r,y+r), fill=(0,128,255), outline=(0,128,255))
    return np.asarray(pil, dtype=np.uint8)