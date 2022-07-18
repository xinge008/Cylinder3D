## Description
[Spconv-V2](https://github.com/traveller59/spconv) made major changes to the convolution operation and how weights are read. [Spconv-V1](https://github.com/traveller59/spconv/tree/v1.2.1) is not supported anymore. Following an unsuccessfull effort to restructure the weights, Cylinder3D was retrained on SemanticKITTI to produced new Spconv-v2 weights (FP32).

This repository is forked from the [original implementation](https://github.com/xinge008/Cylinder3D), with the following major changes:
- Network code updated to Spconv-V2. Credit here goes to the code by @min2209 in[Issue](https://github.com/xinge008/Cylinder3D/issues/107).
- Changed training schedule to CosineAnnealingLR and the optimizer to AdamW with the default weight decay of 1e-2. This is because the validation results of the original authors could not be reproduced with their training script. mIOU in the retrained version is now 63.5 compared to 65.9 in the Paper. Its is likely that further improvements can be made with a more careful choice of hyperparameters when training.

All credit for this repositiory goes to the original authors of Cylinder3D, and additionally for the updated code in /network/segmentator_3d_asymm_spconv.py to @min2209. This repositiory was created to encourage further research and ease of use.

### Issue
Typo in converted model. This affects the DDCM block and its weights, however barely affects performance. [Ref. Issue](https://github.com/xinge008/Cylinder3D/issues/107). This is not updated yet to maintain usability of the weights.
```
reaA = resA.replace_feature(self.bn1(resA.features)) should be: resA = resA.replace_feature(self.bn1(resA.features))
```

## Installation

### Weights
The weights with mIOU 63.5 (Validation, vs 65.9 Original) can be downloaded [here](https://drive.google.com/drive/folders/1LBCRHz2VyeSz4M27GiqhoRuzlKyFvbo1?usp=sharing) and should be placed into the ./network folder.

Weights are trained according to the original Cylinder3D, and not according to knowledge distillation (Cylinder3D PVKD).

### Training
- 40 epochs
- 0.001 LR
- AdamW with default Weight Decay 0.01
- CosineDecay Schedule
- Batch Size 2

### Requirements
The version number is not a requirement, rather the version I have used. I used Ubuntu 20.04 LTS.
- Python 3.8
- PyTorch == 1.11.0
- yaml == 6.0
- strictyaml == 1.6.1
- Cython == 0.29.30
- tqdm == 4.64.0
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) == cu113*
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv-cu113 == 2.1.22](https://github.com/traveller59/spconv) (different CUDA versions available)
- numba == 0.55.2 (install last, as this will likely downgrade your numpy version automatically)

*comment : There is a conda installation possibility. However I had trouble with the anaconda environment and CUDA. Instead I built this package outside of my virtual environment, in the base environment, then copied it manually into the virtual environment.

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

## Training
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running "sh train.sh"

### Training for nuScenes
Please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)
No pretrained weights available.

### Pretrained Models
-- We provide a pretrained model for SemanticKITTI


## Semantic segmentation demo for a folder of lidar scans
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--demo-label-folder is optional
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --demo-label-folder YOUR_LABEL_FOLDER
```

## TODO List
- [x] Release pretrained model for nuScenes.
- [x] Support multiscan semantic segmentation.
- [ ] Support more models, including PolarNet, RandLA, SequeezeV3 and etc.
- [ ] Integrate LiDAR Panotic Segmentation into the codebase.

## Reference

If you find this work useful in your research, please consider citing the [paper](https://arxiv.org/pdf/2011.10033):
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}

#for LiDAR panoptic segmentation
@article{hong2020lidar,
  title={LiDAR-based Panoptic Segmentation via Dynamic Shifting Network},
  author={Hong, Fangzhou and Zhou, Hui and Zhu, Xinge and Li, Hongsheng and Liu, Ziwei},
  journal={arXiv preprint arXiv:2011.11964},
  year={2020}
}
```
