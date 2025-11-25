
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import io
from tqdm import tqdm

import cfg
from utils import *
import pandas as pd
from sklearn.cluster import KMeans
from multirater_metrics import compute_metrics

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).cuda(device=GPUdevice)*2)
seed = torch.randint(1,11,(args.b,7))
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def random_prompt_from_current_mask(
    current_mask: Optional[torch.Tensor],  # (B,1,H,W) or (B,H,W) or None
    multi_rater: torch.Tensor,             # (B,R,H,W) or (B,R,1,H,W) or (R,H,W)/(R,1,H,W)
    prev_rater: Optional[Union[torch.Tensor, List[int]]] = None,  # (B,) or None
    randomize_prob: float = 0.2,
    valid_raters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch version. One click per sample; no posterior/state.
    If current_mask is None, soft GT = mean(multi_rater) and no click is created.
    Returns:
      point_coords:(B,2) float32 (y,x)  (empty when current_mask is None)
      point_labels:(B,)   int64
      gt_soft     :(B,1,H,W) float32
      rater_index :(B,)   int64 chosen rater per sample (or -1 if current_mask is None)
    """
    # ---- normalize multi_rater -> (B,R,H,W) ----
    if multi_rater.dim() == 3:
        multi_rater = multi_rater.unsqueeze(0)      # (1,R,H,W)
    if multi_rater.dim() == 5:
        if multi_rater.size(2) != 1:
            raise ValueError("multi_rater 5D must have channel=1.")
        multi_rater = multi_rater[:, :, 0]          # (B,R,H,W)
    if multi_rater.dim() != 4:
        raise ValueError("multi_rater must be (B,R,H,W) or (B,R,1,H,W) or (R,H,W).")

    B, R, H, W = multi_rater.shape
    device = multi_rater.device
    raters_bool = (multi_rater > 0.5)     # (B,R,H,W)
    raters_f    = raters_bool.float()
    
    if valid_raters is None:
        valid = torch.ones((B, R), dtype=torch.bool, device=device)
    else:
        valid = valid_raters.to(device=device).bool()
        if valid.dim() == 1:  # (R,) -> (1,R) -> (B,R)
            valid = valid.unsqueeze(0).expand(B, R)
        assert valid.shape == (B, R)

    # If no current prediction yet: return soft mean and no click
    if current_mask is None:
        gt_soft = raters_f.mean(dim=1, keepdim=True)            # (B,1,H,W)
        point_coords = torch.empty((B, 2), dtype=torch.float32, device=device)
        point_labels = torch.empty((B,),   dtype=torch.int64,   device=device)
        rater_index  = torch.full((B,), -1, dtype=torch.int64,  device=device)
        return point_coords, point_labels, gt_soft, rater_index

    # --- normalize current_mask if provided -> (B,1,H,W) probs ---
    if current_mask is None:
        cur_prob = None
    else:
        if current_mask.dim() == 2:
            current_mask = current_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif current_mask.dim() == 3:
            # (B,H,W) -> (B,1,H,W)  OR  (1,H,W) -> (B,1,H,W)
            if current_mask.size(0) == B:
                current_mask = current_mask.unsqueeze(1)
            elif current_mask.size(0) == 1:
                current_mask = current_mask.unsqueeze(1).expand(B, 1, H, W)
            else:
                current_mask = current_mask.unsqueeze(0).unsqueeze(0)
        elif current_mask.dim() == 4 and current_mask.size(1) != 1:
            raise ValueError("current_mask must have channel=1 if 4D.")
        cur_prob = current_mask.float()

    # --- choose rater indices (valid-only), possibly follow prev_rater ---
    if isinstance(prev_rater, list):
        prev_vec = torch.tensor(prev_rater, device=device, dtype=torch.long)
    elif isinstance(prev_rater, torch.Tensor):
        prev_vec = prev_rater.to(device=device, dtype=torch.long)
    else:
        prev_vec = torch.full((B,), -1, device=device, dtype=torch.long)

    chosen_idx = torch.empty((B,), dtype=torch.long, device=device)
    for b in range(B):
        valid_idx = torch.nonzero(valid[b], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            # fallback: consider all
            valid_idx = torch.arange(R, device=device)
        pv = int(prev_vec[b].item())
        follow_prev = (pv >= 0) and (pv < R) and (valid[b, pv].item())
        if follow_prev and (torch.rand(1, device=device).item() >= randomize_prob):
            chosen_idx[b] = pv
        else:
            # sample uniformly from valid set (exclude prev if it was valid and we decided to switch)
            if follow_prev and valid_idx.numel() > 1:
                pool = valid_idx[valid_idx != pv]
            else:
                pool = valid_idx
            j = torch.randint(0, pool.numel(), (1,), device=device).item()
            chosen_idx[b] = int(pool[j].item())

    # --- STEP-0: no click, soft GT = mean over *valid* raters ---
    if cur_prob is None:
        # weighted mean over valid raters: sum(valid * raters) / sum(valid)
        w = valid.float().unsqueeze(-1).unsqueeze(-1)                     # (B,R,1,1)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)                # (B,1,1,1)
        gt_soft = (raters_f * w).sum(dim=1, keepdim=True) / denom         # (B,1,H,W)
        point_coords = torch.empty((B, 0, 2), dtype=torch.float32, device=device)
        point_labels = torch.empty((B, 0),    dtype=torch.int64,   device=device)
        return point_coords, point_labels, gt_soft, chosen_idx

    # --- later steps: pick one click per item from disagreement wrt chosen valid rater ---
    y = torch.empty((B,), dtype=torch.long, device=device)
    x = torch.empty((B,), dtype=torch.long, device=device)
    lbl = torch.empty((B,), dtype=torch.long, device=device)

    for b in range(B):
        r_mask_b = raters_bool[b, chosen_idx[b]]       # (H,W) bool
        r_mask_f = r_mask_b.float()
        diff_mag = (cur_prob[b, 0] - r_mask_f).abs().view(-1)  # (H*W)
        k = min(20, diff_mag.numel())
        top_vals, top_idx = torch.topk(diff_mag, k, largest=True)
        j = torch.randint(0, k, (1,), device=device).item()
        flat = int(top_idx[j].item())
        y[b] = flat // W
        x[b] = flat %  W
        lbl[b] = int(r_mask_b[y[b], x[b]].item())      # {0,1} -> supports negative prompt

    # soft GT = normalized mean over *valid & agreeing* raters
    raters_BHW_R = raters_bool.permute(0, 2, 3, 1)                         # (B,H,W,R)
    raters_px    = raters_BHW_R[torch.arange(B, device=device), y, x]      # (B,R) bool
    matches      = (raters_px == lbl.unsqueeze(1)) & valid                  # (B,R) bool
    w = matches.float()
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)                     # (B,1)
    gt_soft = (raters_f * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).unsqueeze(1) / denom.unsqueeze(-1).unsqueeze(-1)

    point_coords = torch.stack([y, x], dim=1).unsqueeze(1).to(dtype=torch.float32)  # (B,1,2) (y,x)
    point_labels = lbl.unsqueeze(1)                                                  # (B,1)
    return point_coords, point_labels, gt_soft, chosen_idx

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, runs=6):
    epoch_loss = 0
    ind = 0

    # train mode
    net.train()
    optimizer.zero_grad()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)

            multi_rater = pack['multi_rater'].to(dtype = torch.float32, device = GPUdevice)
            if multi_rater.dim() == 5 and multi_rater.size(2) == 1:
                multi_rater = multi_rater[:, :, 0]
            valid_raters = pack['valid_raters'].to(device = GPUdevice)

            B, _, H, W = imgs.shape
            coords_torch = torch.empty((B, 0, 2), dtype=torch.float32, device=GPUdevice)  # (y,x)
            labels_torch = torch.empty((B, 0),    dtype=torch.int64,   device=GPUdevice)

            weights   = torch.tensor(net.EM_weights.weights,   dtype=torch.float, device=GPUdevice)
            means     = torch.tensor(net.EM_mean_variance.means,     dtype=torch.float, device=GPUdevice)
            variances = torch.tensor(net.EM_mean_variance.variances, dtype=torch.float, device=GPUdevice)
            pred_masks_weights_list = weights.unsqueeze(0).repeat(B, 1)                  # (B, n_components)

            last_pred = None
            prev_rater_vec = torch.full((B,), -1, device=GPUdevice, dtype=torch.long)
            
            _, _, gt_current, prev_rater_vec = random_prompt_from_current_mask(
                current_mask=None,
                multi_rater=multi_rater,
                prev_rater=prev_rater_vec,
                randomize_prob=0.2,
                valid_raters=valid_raters,
            )

            for run in range(runs):              
                '''Train image encoder, combine net(inside image encoder), mask decoder'''
                # prompt encoder
                for n, value in net.prompt_encoder.named_parameters(): 
                    value.requires_grad = False
                se, de = net.prompt_encoder( 
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice)

                # EM_mean_variance
                for n, value in net.EM_mean_variance.named_parameters(): 
                    value.requires_grad = False
                means, variances = net.EM_mean_variance(se, pe)

                # EM_weights
                for n, value in net.EM_weights.named_parameters(): 
                    value.requires_grad = False
                weights= net.EM_weights(pred_masks_weights_list)
                weights = weights.mean(axis=0).to(device = GPUdevice)
                weights /= weights.sum() 
                
                # image encoder & combine net
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                imge_list = net.image_encoder(imgs, weights, means, variances, num_sample=1)            
                
                # mask decoder
                for n, value in net.mask_decoder.named_parameters(): 
                    value.requires_grad = True

                pred_list_last_pred = []
                pred_list_image_size = []
                pred_list_output_size = []

                pred, _ = net.mask_decoder(
                    image_embeddings=imge_list[0],
                    image_pe=pe, 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
                # for last_pred
                pred_list_last_pred.append(pred)

                # Resize to the image size
                pred_image_size = F.interpolate(pred,size=(args.image_size, args.image_size)) 
                # standardlise before cluster
                if torch.max(pred_image_size) > 1 or torch.min(pred_image_size) < 0:
                    pred_image_size = torch.sigmoid(pred_image_size)
                pred_list_image_size.append(pred_image_size)

                # Resize to the output size
                pred_output_size = F.interpolate(pred,size=(args.image_size, args.image_size))
                pred_list_output_size.append(pred_output_size)


                # result for last_pred
                pred_list_last_pred = torch.stack(pred_list_last_pred, dim=0)
                last_pred = torch.mean(pred_list_last_pred, dim=0).detach().clone()

                # result for output_size
                pred_list_output_size = torch.stack(pred_list_output_size, dim=0)
                output = torch.mean(pred_list_output_size, dim=0)

                #pred_list_output = (pred_list_output> 0.5).float()
                loss = lossfunc(output, gt_current)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                pred_list_image_size = [ torch.sigmoid(t) if (t.max()>1 or t.min()<0) else t
                                         for t in pred_list_image_size ]
                select_list_mean = torch.stack(pred_list_image_size, dim=0).mean(dim=0).detach().clone()  # (B,1,H,W)

                pt_next, pl_next, gt_soft_next, r_idx = random_prompt_from_current_mask(
                    current_mask=select_list_mean,
                    multi_rater=multi_rater,
                    prev_rater=prev_rater_vec,
                    randomize_prob=0.2,
                    valid_raters=valid_raters,
                )
                prev_rater_vec = r_idx

                if pt_next.size(1) > 0:
                    coords_torch = torch.cat((coords_torch, pt_next), dim=1)   # (B,T+1,2)  (y,x)
                    labels_torch = torch.cat((labels_torch, pl_next), dim=1)


                # calculate current weights (output size)
                pred_masks_weights_list = []
                for i in range(imgs.size(0)):
                    pred_masks_weights_list.append(net.EM_weights.compute_weights(
                        torch.flatten(select_list_mean[i].detach().clone()), weights, means, variances))
                pred_masks_weights_list = torch.stack(pred_masks_weights_list, dim=0) #(batch_size, n_components)
                            

                #-----------------------------------------------------------------------

                """train prompt_encoder & EM_mean_variance"""
                # prompt encoder
                for n, value in net.prompt_encoder.named_parameters(): 
                    value.requires_grad = True
                se, de = net.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice) 

                # EM_mean_variance
                for n, value in net.EM_mean_variance.named_parameters(): 
                    value.requires_grad = True
                means, variances = net.EM_mean_variance(se, pe)

                # EM_weights
                for n, value in net.EM_weights.named_parameters(): 
                    value.requires_grad = False
                weights= net.EM_weights(pred_masks_weights_list)
                weights = weights.mean(axis=0).to(device = GPUdevice)
                weights /= weights.sum()

                # image encoder & combine net
                gt_target_em = gt_soft_next.detach()

                for _, v in net.EM_mean_variance.named_parameters(): 
                    v.requires_grad = False
                means, variances = net.EM_mean_variance(se, pe)

                pred_masks_weights_list_train = []
                true_masks_weights_list_train = []
                for i in range(imgs.size(0)):
                    pred_masks_weights_list_train.append(
                        net.EM_weights.compute_weights(
                            torch.flatten(select_list_mean[i]),
                            weights, means, variances
                        )
                    )
                    true_masks_weights_list_train.append(
                        net.EM_weights.compute_weights(
                            torch.flatten(gt_target_em[i]),
                            weights, means, variances
                        )
                    )
                pred_masks_weights_list_train = torch.stack(pred_masks_weights_list_train, dim=0)
                true_masks_weights_list_train = torch.stack(true_masks_weights_list_train, dim=0)

                for _, v in net.EM_weights.named_parameters(): 
                    v.requires_grad = True
                updated_weights_list = net.EM_weights(pred_masks_weights_list_train)
                loss = nn.MSELoss()(updated_weights_list, true_masks_weights_list_train)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # update soft gt for next run
                gt_current = gt_soft_next

            pbar.update()

    return epoch_loss/len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module, runs=6, selected_rater_df_path = False):
    # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    total_loss_list = np.zeros(runs)
    total_eiou_list = np.zeros(runs)
    total_dice_list = np.zeros(runs)
    
    multi_names = ["GED","Dice_max","Dice_match","Dice_soft","Dice_personal_mean","Dice_follow","Delta_follow_vs_neutral"]
    multi_sums_by_run = {r: {k: 0.0 for k in multi_names} for r in range(runs)}
    multi_cnt_by_run  = {r: 0 for r in range(runs)}

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:

        for ind, pack in enumerate(val_loader):
            
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)

            multi_rater = pack['multi_rater'].to(dtype = torch.float32, device = GPUdevice)
            if multi_rater.dim() == 5 and multi_rater.size(2) == 1:
                multi_rater = multi_rater[:, :, 0]
            valid_raters = pack['valid_raters'].to(device = GPUdevice)

            B, _, H, W = imgs.shape
            
            raters_f = (multi_rater > 0.5).float()                                       # (B,R,H,W)
            w = valid_raters.float().unsqueeze(-1).unsqueeze(-1)                          # (B,R,1,1)
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)                            # (B,1,1,1)
            neutral_soft = (raters_f * w).sum(dim=1, keepdim=True) / denom                # (B,1,H,W)
            masks = (neutral_soft > 0.5).float()                                          # (B,1,H,W), binary
            masks_all = masks.unsqueeze(1).repeat(1, runs, 1, 1, 1)                       # (B,runs,1,H,W)
            
            coords_torch = torch.empty((B, 0, 2), dtype=torch.float32, device=GPUdevice)  # (y,x)
            labels_torch = torch.empty((B, 0),    dtype=torch.int64,   device=GPUdevice)

            weights   = torch.tensor(net.EM_weights.weights,   dtype=torch.float, device=GPUdevice)
            means     = torch.tensor(net.EM_mean_variance.means,     dtype=torch.float, device=GPUdevice)
            variances = torch.tensor(net.EM_mean_variance.variances, dtype=torch.float, device=GPUdevice)
            pred_masks_weights_list = weights.unsqueeze(0).repeat(B, 1)                  # (B, n_components)

            last_pred = None
            prev_rater_vec = torch.full((B,), -1, device=GPUdevice, dtype=torch.long)
            
            _, _, gt_current, prev_rater_vec = random_prompt_from_current_mask(
                current_mask=None,
                multi_rater=multi_rater,
                prev_rater=prev_rater_vec,
                randomize_prob=0.0,
                valid_raters=valid_raters,
            )
            
            preds_over_runs = []
            for run in range(runs):
                for _, v in net.prompt_encoder.named_parameters():
                    v.requires_grad = False
                se, de = net.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device=GPUdevice)

                for _, v in net.EM_mean_variance.named_parameters():
                    v.requires_grad = False
                means, variances = net.EM_mean_variance(se, pe)
                    
                for _, v in net.EM_weights.named_parameters():
                    v.requires_grad = False
                weights = net.EM_weights(pred_masks_weights_list)
                weights = weights.mean(axis=0).to(device=GPUdevice)
                weights /= weights.sum()

                # -- Image encoder --
                for _, v in net.image_encoder.named_parameters():
                    v.requires_grad = False
                imge_list = net.image_encoder(imgs, weights, means, variances, num_sample=1)

                for _, v in net.mask_decoder.named_parameters():
                    v.requires_grad = False

                pred, _ = net.mask_decoder(
                    image_embeddings=imge_list[0],
                    image_pe=pe,
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=(args.multimask_output > 1),
                )
                last_pred = pred.detach().clone() 


                # Resize: image_size (probs) & output_size (logits)
                pred_image_size = F.interpolate(pred, size=(args.image_size, args.image_size))
                if (pred_image_size.max() > 1) or (pred_image_size.min() < 0):
                    pred_image_size = torch.sigmoid(pred_image_size)                      # to [0,1] for click
                pred_output_size = F.interpolate(pred, size=(args.image_size, args.image_size))
                output_logits = pred_output_size                                          # (B,1,H,W)
                output_bin = (torch.sigmoid(output_logits) > 0.5).float()                 # for eval_seg
                preds_over_runs.append(output_logits.detach().clone())

                # ---- Metrics & loss per-run ----
                temp = eval_seg(output_bin, masks_all[:, run], threshold)
                loss = criterion_G(output_logits, masks_all[:, run])

                total_eiou_list[run] += temp[0]
                total_dice_list[run] += temp[1]
                total_loss_list[run] += float(loss.item())
                
                # Compute multi-rater metrics
                
                pred_set_sofar = torch.stack(preds_over_runs, dim=1)  # (B,M<=run+1,1,H,W)
                follow_idx_for_run = prev_rater_vec if (prev_rater_vec.min() >= 0) else None
                
                multirater_metrics =  compute_metrics(
                        preds_single=output_logits,
                        raters=multi_rater,                 # (B,R,H,W) ok
                        valid_raters=valid_raters,
                        pred_set=pred_set_sofar,
                        follow_idx=follow_idx_for_run,
                        dice_match_norm="gt",
                    )       

                for k in multi_names:
                    multi_sums_by_run[run][k] += float(multirater_metrics[k])
                multi_cnt_by_run[run] += 1
                
                # Update next prompt
                
                select_list_mean = pred_image_size.detach().clone()                        # (B,1,H,W) probs
                pt_next, pl_next, gt_soft_next, r_idx = random_prompt_from_current_mask(
                    current_mask=select_list_mean,
                    multi_rater=multi_rater,
                    prev_rater=prev_rater_vec,
                    randomize_prob=0.0,
                    valid_raters=valid_raters,
                )
                prev_rater_vec = r_idx
                if pt_next.size(1) > 0:
                    coords_torch = torch.cat((coords_torch, pt_next), dim=1)              # (B,T+1,2) (y,x)
                    labels_torch = torch.cat((labels_torch, pl_next), dim=1)              

                pred_masks_weights_list = []
                for i in range(B):
                    pred_masks_weights_list.append(
                        net.EM_weights.compute_weights(
                            torch.flatten(select_list_mean[i].detach().clone()),
                            weights, means, variances
                        )
                    )
                pred_masks_weights_list = torch.stack(pred_masks_weights_list, dim=0)
            
            
                
            ind += 1
            pbar.set_postfix(
                dice=float(total_dice_list[-1] / max(1, ind)),
                eiou=float(total_eiou_list[-1] / max(1, ind)),
                loss=float(total_loss_list[-1] / max(1, ind)),
            )
            pbar.update()

    total_loss_list = [v / max(1, ind) for v in total_loss_list]
    total_eiou_list = [v / max(1, ind) for v in total_eiou_list]
    total_dice_list = [v / max(1, ind) for v in total_dice_list]
    
    multi_avg_by_run = []
    for r in range(runs):
        rec = {"run": r}
        denom = max(1, multi_cnt_by_run[r])
        for k in multi_names:
            rec[k] = multi_sums_by_run[r][k] / denom
        multi_avg_by_run.append(rec)

    return total_loss_list, (total_eiou_list, total_dice_list), multi_avg_by_run

