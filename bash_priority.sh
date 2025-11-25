python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -mode SPA

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset LIDC -data_path './data/LIDC' -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg -data_path './data/MBHSeg-Binary' -mode SPA

python train.py -exp_name 'MBHSeg_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset MBHSeg -data_path './data/MBHSeg-Binary' -mode SPA

python train.py -exp_name 'LIDC_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -mode AESPA

python train.py -exp_name 'LIDC_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset LIDC -data_path './data/LIDC' -mode AESPA

python train.py -exp_name 'MBHSeg_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg -data_path './data/MBHSeg-Binary' -mode AESPA

python train.py -exp_name 'MBHSeg_AESPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -b 64 -dataset MBHSeg -data_path './data/MBHSeg-Binary' -mode AESPA