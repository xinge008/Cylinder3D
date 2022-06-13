## Description
[Spconv-V2](https://github.com/traveller59/spconv) made major changes to the convolution operation and how weights are read. [Spconv-V1](https://github.com/traveller59/spconv/tree/v1.2.1) is not supported anymore. For this reason an unsuccessfull effort was made to restructure the weights of Cylinder3D-spconv-V1 to spconv-V2, see [Issue](https://github.com/xinge008/Cylinder3D/issues/107).

This repository contains the retrained Cylinder3D Network on Semantic-KITTI using spconv-V2, based on the updated code by @min2209. All credit for this repositiory goes to the original authors of Cylinder3D, and additionally for the updated code in /network/segmentator_3d_asymm_spconv.py to @min2209. This repositiory was created to encourage further research and ease of use.

## Installation

### Weights
Weights can be downloaded [here](https://drive.google.com/drive/folders/1LBCRHz2VyeSz4M27GiqhoRuzlKyFvbo1?usp=sharing) and should be placed into the ./weights folder.

### Requirements
- PyTorch >= 1.2 
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

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
