# !/bin/bash

# SPA ablations
python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 16 -dataset LIDC -data_path './data/LIDC' -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 24 -n_clusters 4 -n_components 16 -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 12 -n_clusters 4 -n_components 16 -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 3 -n_components 16 -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 16 -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 4 -n_components 8 -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 4 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 16 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 24 -n_clusters 4 -n_components 16 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 12 -n_clusters 4 -n_components 16 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 3 -n_components 16 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 2 -n_components 16 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 4 -n_components 8 -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 2 -n_components 4 -mode SPA
# AESPA runs
python train.py -exp_name 'LIDC_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -mode AESPA

python train.py -exp_name 'LIDC_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset LIDC -data_path './data/LIDC' -mode AESPA

python train.py -exp_name 'MBHSeg_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -mode AESPA

python train.py -exp_name 'MBHSeg_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -mode AESPA

