python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC'

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC'

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 256 -out_size 256 -b 16 -dataset LIDC -data_path './data/LIDC'

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 4 -n_components 16

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 24 -n_clusters 4 -n_components 16

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 12 -n_clusters 4 -n_components 16

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 3 -n_components 16

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 16

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 4 -n_components 8

python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 2 -n_components 4