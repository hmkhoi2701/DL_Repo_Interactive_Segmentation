import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from PIL import Image
from IPython.display import display
import cv2
from tqdm import tqdm
import os

def apply_window(arr, level = 40, width = 80):
    lo = level - width / 2.0
    hi = level + width / 2.0
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

for split in ['train', 'test']:
    os.makedirs(f'data/MBHSeg-Binary/{split}/imgs', exist_ok=True)
    os.makedirs(f'data/MBHSeg-Binary/{split}/gt', exist_ok=True)
    os.makedirs(f'data/MBHSeg-Multi/{split}/imgs', exist_ok=True)
    os.makedirs(f'data/MBHSeg-Multi/{split}/gt', exist_ok=True)
    for imgname in tqdm(os.listdir(f'data/MBHSeg/{split}/')):
        folder = f'data/MBHSeg/{split}/' + imgname
        multiraters = []
        img_slices = []
        keep_slices = []
        for filename in sorted(os.listdir(folder)):
            if 'image' in filename:
                img = nib.load(os.path.join(folder, filename)).get_fdata()
                n_slices = img.shape[2]
                for i in range(n_slices):
                    slice_img = img[:,:,i]
                    slice_img = apply_window(slice_img)
                    slice_img = (slice_img * 255).astype(np.uint8)
                    img_slices.append(slice_img)
            else:
                rater_id = filename.split('.')[0][-1]
                mask = nib.load(os.path.join(folder, filename)).get_fdata()
                multiraters.append((mask,rater_id))
            
        for i in range(n_slices):
            reject = True
            for rater in range(len(multiraters)):
                slice_mask = multiraters[rater][0][:,:,i]
                if np.sum(slice_mask) != 0:
                    reject = False
                    break
            if reject == False:
                keep_slices.append(i)
                
        if len(keep_slices) == 0:
            continue
        
        os.makedirs(os.path.join(f'data/MBHSeg-Binary/{split}/imgs/{imgname}'), exist_ok=True)
        os.makedirs(os.path.join(f'data/MBHSeg-Multi/{split}/imgs/{imgname}'), exist_ok=True)
        os.makedirs(os.path.join(f'data/MBHSeg-Binary/{split}/gt/{imgname}'), exist_ok=True)
        os.makedirs(os.path.join(f'data/MBHSeg-Multi/{split}/gt/{imgname}'), exist_ok=True)

        for slice_idx in keep_slices:
            
            img_slice = img_slices[slice_idx]
            cv2.imwrite(os.path.join(f'data/MBHSeg-Binary/{split}/imgs/{imgname}', 'slice_' + str(slice_idx) + '.png'), img_slice)
            cv2.imwrite(os.path.join(f'data/MBHSeg-Multi/{split}/imgs/{imgname}', 'slice_' + str(slice_idx) + '.png'), img_slice)
            
            for rater in range(len(multiraters)):
                rater_id = multiraters[rater][1]
                slice_mask = multiraters[rater][0][:,:,slice_idx]
                slice_binary_mask = (slice_mask > 0).astype(np.uint8) * 255
                slice_one_hot_mask = np.zeros((slice_mask.shape[0], slice_mask.shape[1], 6), dtype=np.uint8)
                for cls in range(6):
                    slice_one_hot_mask[:,:,cls] = (np.abs(slice_mask - float(cls)) < 1e-6).astype(np.uint8) * 255
                np.save(os.path.join(f'data/MBHSeg-Binary/{split}/gt/{imgname}', 'slice_' + str(slice_idx) + '_r' + rater_id + '.npy'), slice_binary_mask)
                np.save(os.path.join(f'data/MBHSeg-Multi/{split}/gt/{imgname}', 'slice_' + str(slice_idx) + '_r' + rater_id + '.npy'), slice_one_hot_mask)