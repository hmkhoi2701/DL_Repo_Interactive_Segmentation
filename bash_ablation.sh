# CUDA_VISIBLE_DEVICES=0 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_1' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 24 -n_clusters 4 -n_components 16 -mode SPA >> log_exp_priority_1.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_2' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 12 -n_clusters 4 -n_components 16 -mode SPA >> log_exp_priority_2.txt &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_3' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 3 -n_components 16 -mode SPA >> log_exp_priority_3.txt &

# CUDA_VISIBLE_DEVICES=3 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_4' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 16 -mode SPA >> log_exp_priority_4.txt &

# CUDA_VISIBLE_DEVICES=4 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_5' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 4 -n_components 8 -mode SPA >> log_exp_priority_5.txt &

# CUDA_VISIBLE_DEVICES=5 nohup python train.py -exp_name 'LIDC_SPA_exp_ablation_6' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 4 -mode SPA >> log_exp_priority_6.txt &

# CUDA_VISIBLE_DEVICES=6 nohup python train.py -exp_name 'MBHSeg_SPA_exp_ablation_7' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 24 -n_clusters 4 -n_components 16 -mode SPA >> log_exp_priority_7.txt &

# CUDA_VISIBLE_DEVICES=7 nohup python train.py -exp_name 'MBHSeg_SPA_exp_ablation_8' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 12 -n_clusters 4 -n_components 16 -mode SPA >> log_exp_priority_8.txt &

CUDA_VISIBLE_DEVICES=0 python train.py -exp_name 'MBHSeg_SPA_exp_ablation_9' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 3 -n_components 16 -mode SPA >> log_exp_priority_9.txt &

CUDA_VISIBLE_DEVICES=1 python train.py -exp_name 'MBHSeg_SPA_exp_ablation_10' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 2 -n_components 16 -mode SPA >> log_exp_priority_10.txt & 

CUDA_VISIBLE_DEVICES=2 python train.py -exp_name 'MBHSeg_SPA_exp_ablation_11' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 4 -n_components 8 -mode SPA >> log_exp_priority_11.txt &

CUDA_VISIBLE_DEVICES=3 python train.py -exp_name 'MBHSeg_SPA_exp_ablation_12' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -b 64 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 2 -n_components 4 -mode SPA >> log_exp_priority_12.txt &