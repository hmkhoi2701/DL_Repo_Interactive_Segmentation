<h1 align="center">● SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation</h1>

SPA, is an advanced segmentation framework that efficiently adapts to diverse test-time preferences with minimal human interaction. By presenting users a select few, distinct segmentation candidates that best capture uncertainties, it reduces clinician workload in reaching the preferred segmentation. This method is elaborated on the paper:

[SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation](https://arxiv.org/abs/2411.15513) (ICCV 2025)

and [SPA webpage](https://ImprintLab.github.io/SPA/).

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="400" src="https://github.com/SuperMedIntel/SPA/blob/main/static/assets/images/facial.png"></div>
Our uncertainty-aware interactive segmentation model, <b>SPA</b>, efficiently achieves segmentations whose decisions on uncertain pixels are aligned with users preferences. This is achieved by modeling uncertainties and human interactions. At inference time, users are presented with one recommended prediction and a few representative segmentations that capture uncertainty, allowing users to select the one best aligned with their clinical needs. If the user is unsatisfied with the recommended prediction, the model learns from the users' selections, adapts itself, and presents users a new set of representative segmentations. Our approach minimizes user interactions and eliminates the need for painstaking pixel-wise adjustments compared to conventional interactive segmentation models.

## 🧐 Requirement

 Install the environment:

 ``conda create -n seg_environment python=3.10``

 ``conda activate seg_environment``

 ``pip install -r requirements.txt``

## 🎯 Example Cases
**Step1:** Download MedSAM pretrained weight and put in the ``./checkpoint/`` folder, create the folder if it does not exist ⚒️
 [MedSAM checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) -> Select `medsam_vit_b.pth` file.
 

**Step2:** Download datasets to `data` folder:
- [MBHSeg](https://www.mbhseg.com/)
- LIDC-IDRI from GCS commands in [this notebook](https://colab.research.google.com/github/deepmind/deepmind-research/blob/master/hierarchical_probabilistic_unet/HPU_Net.ipynb#scrollTo=SY_lyR2BHRu9)
- [REFUGE](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/tree/main)

Then run `process_mbh.py` to process MBHSeg dataset. The final structure should look like this
```
data/
├── LIDC/
│   ├── train/
│   │   ├── gt/
│   │   │   ├──LIDC-IDRI-0001/
│   │   │   │   ├──z-105.0_c0_l0.png
│   │   │   │   ├──z-105.0_c0_l1.png
│   │   │   │   └──...
│   │   │   └── ...
│   │   └── images/
│   │       ├──LIDC-IDRI-0001/
│   │       │   ├──z-105.0_c0.png
│   │       │   └──...
│   │       └── ...
│   └── test/...
│
├── MBHSeg-Binary/
│   ├── train/
│   │   ├── gt/
│   │   │   ├──ID_0b10cbee_ID_f91d6a7cd2/
│   │   │   │   ├──slice_6_r1.npy
│   │   │   │   ├──slice_6_r3.npy
│   │   │   │   └──...
│   │   └── images/
│   │       ├──ID_0b10cbee_ID_f91d6a7cd2/
│   │       │   ├──slice_6.png
│   │       │   └──...
│   │       └── ...
│   └── test/...
├── MBHSeg-Multi/...
│
└── REFUGE-Multirater/
    ├── Training-400/
    │   ├── 0826/
    │   │   ├── 0826.jpg
    │   │   ├── 0826_seg_cup_1.png
    │   │   ├── 0826_seg_cup_2.png
    │   │   └──...       
    │   └── ...
    └── Test-400/...
 ```

 Or alternatively, simply download from our huggingface [dataset](https://huggingface.co/datasets/bampyeonji/multiraters/tree/main) and unzip.


**Step3:** Run the training by:
 <!-- ``python train.py -net sam -mod sam -exp_name 'REFUGE_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 512 -out_size 256 -b 4 -val_freq 1 -dataset REFUGE -data_path './data/REFUGE'`` -->

``python train.py -exp_name 'LIDC_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset LIDC -data_path './data/LIDC' -num_samples 48 -n_clusters 4 -n_components 16``

``python train.py -exp_name 'MBHSeg_Binary_SPA' -sam_ckpt ./checkpoint/medsam_vit_b.pth -image_size 128 -out_size 128 -b 16 -dataset MBHSeg-Binary -data_path './data/MBHSeg-Binary' -num_samples 48 -n_clusters 4 -n_components 16``

Other configs to try are listed in `bash.sh`

<!-- **Step4:** Run the validation by:
 ``python val.py -net sam -mod sam -exp_name 'val' -vis 1 -sam_ckpt CHECKPOINT_PATH -weights CHECKPOINT_PATH -image_size 512 -out_size 256 -b 1 -val_freq 1 -dataset REFUGE -data_path './data/REFUGE'`` -->

## Credits
 ~~~
@misc{zhu_spa_2024,
      title={SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation},
      author={Zhu, Jiayuan and Wu, Junde and Ouyang, Cheng and Kamnitsas, Konstantinos and Noble, Alison},
      url = {http://arxiv.org/abs/2411.15513},
      doi = {10.48550/arXiv.2411.15513},
      year = {2024},
    }
 ~~~
