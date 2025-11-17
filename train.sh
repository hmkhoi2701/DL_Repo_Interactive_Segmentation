python train.py -exp_name 'LIDC_SPA_new' -sam_ckpt ./checkpoint/medsam_vit_b.pth \
                -image_size 128 -out_size 128 -b 16 -val_freq 1 -dataset LIDC -data_path './data/LIDC'