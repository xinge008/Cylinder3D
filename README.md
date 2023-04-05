## Spconv
Some issues can occur with newer versions of Spconv 2.x. If you get an error, that precompiled functions are not found, make sure you do not have duplicates of the package cumm-[cuda version]. If issues persists, spconv version 2.2.3 or lower is recommended.

## TODO
- [ ] PyTorch 2.0 support
- [ ] Removal of torch-scatter for PyTorch native implementation (package functionalities were integrated into PyTorch as .scatter_reduce_)
- [X] Fix issue in converted network (see below)
- [X] Test with Cuda 11.6/11.7 support
- [X] [Spconv 2.2](https://github.com/traveller59/spconv) (FP32 disabled by default, see their instructions on how to enable FP32 if necessary.)

## Description
This repository is forked from the [original implementation](https://github.com/xinge008/Cylinder3D). This repositiory was created to encourage continued research and provide a significantly faster implementation.

[Spconv-V2](https://github.com/traveller59/spconv) made major changes to the convolution operation and how weights are read. [Spconv-V1](https://github.com/traveller59/spconv/tree/v1.2.1) is not supported anymore. Following an unsuccessfull effort to restructure the weights, Cylinder3D was retrained on SemanticKITTI to produced new Spconv-v2 weights.

Note: the version released publicly by the authors is from their [first paper](https://arxiv.org/pdf/2008.01550.pdf) and does not include the pointwise refinement module or weighted cross entropy. The CVPR version has not been made publically available. The mIOU of the retrained Spconv2 Version is 63.2, compared to 64.3 mIOU in page 7 in the paper. The original implementation does not contain manual seeding. To achieve the 63.2 mIOU result, the training regimen had to be changed slightly, as the Paper results could not be reproduced (See Training).

## Improvements
- Network code updated to Spconv-v2.1.x Credit here goes to the code by @min2209 in [Issue](https://github.com/xinge008/Cylinder3D/issues/107).
- Spconv-v2.x receives continued support. Speedup of 50 - 80% compared to original implementation
- Mixed precision support for further speedup during training

It is likely that further improvements can be made with a more careful choice of hyperparameters when training.

### Issus
Fixed Typo in converted model. Weights are updated.
```
reaA = resA.replace_feature(self.bn1(resA.features)) should be: resA = resA.replace_feature(self.bn1(resA.features))
```

## Installation

### Weights
The weights with mIOU 63.2 (Validation, vs 64.3 Original) can be downloaded [here](https://drive.google.com/drive/folders/1LBCRHz2VyeSz4M27GiqhoRuzlKyFvbo1?usp=sharing) and should be placed into the ./network folder.

Weights are trained according to the original Cylinder3D, and not according to knowledge distillation (Cylinder3D PVKD).

### Training
- 40 epochs
- 0.00707 base LR with sqrt_k scaling rule (equals to original 0.001 at batchsize = 2, equals 0.00489 at batchsize = 24)
- AdamW with Weight Decay 0.001
- CosineDecay Schedule
- Batch Size 24 (Better result possible with lower batch size, batch size chosen for economical reasons.)

### Requirements
Also tested with CUDA 11.3, just "downgrade" pytorch, spconv and torch-scatter.

Tested on Ubuntu 20.04 LTS. Recommend pip install over conda install.
- Python 3.8
- PyTorch == 1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
- yaml == 6.0
- strictyaml == 1.6.1
- Cython == 0.29.30
- tqdm == 4.64.0
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) == cu116
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv-cu117 == 2.2.3](https://github.com/traveller59/spconv) (different CUDA versions available)
- numba == 0.55.2 (install last, as this will likely downgrade your numpy version automatically)

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

## Training
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running "sh train.sh"


If you find this work useful in your research, please consider citing the original authors [papers](https://arxiv.org/pdf/2011.10033):
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}
```
