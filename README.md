## UniScene: Unified Occupancy-centric Driving Scene Generation [CVPR 2025]



 [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2412.05435-purple)](https://arxiv.org/abs/2412.05435) 
[![Code page](https://img.shields.io/badge/Project%20Page-UniScene-red)](https://arlo0o.github.io/uniscene/)


### Demo:
<div align=center><img width="960" height="220" src="./assets/teaser_fig1.png"/></div>
**(a) Overview of UniScene.** Given BEV layouts, UniScene facilitates versatile data generation, including semantic occupancy, multi-view video, and LiDAR point clouds, through an occupancy-centric hierarchical modeling approach. **(b) Performance comparison on different generation tasks.** UniScene delivers substantial improvements over SOTA methods in video, LiDAR, and occupancy generation.



<!-- <div align=center><img width="960" height="470" src="./assets/teaser_fig1_b.png"/></div>
 Versatile generation ability of UniScene.  -->

<br>

<div align=center><img width="960" height="540" src="./assets/demo.gif"/></div>



### Abstract:
Generating high-fidelity, controllable, and annotated training data is critical for autonomous driving. Existing methods typically generate a single data form directly from a coarse scene layout, which not only fails to output rich data forms required for diverse downstream tasks but also struggles to model the direct layout-to-data distribution. In this paper, we introduce UniScene, the first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. UniScene employs a progressive generation process that decomposes the complex task of scene generation into two hierarchical steps: (a) first generating semantic occupancy from a customized scene layout as a meta scene representation rich in both semantic and geometric information, and then (b) conditioned on occupancy, generating video and LiDAR data, respectively, with two novel transfer strategies of Gaussian-based Joint Rendering and Prior-guided Sparse Modeling. This occupancy-centric approach reduces the generation burden, especially for intricate scenes, while providing detailed intermediate representations for the subsequent generation stages. Extensive experiments demonstrate that UniScene outperforms previous SOTAs in the occupancy, video, and LiDAR generation, which also indeed benefits downstream driving tasks.


### Framework:
<div align=center><img width="960" height="270" src="./assets/overall.png"/></div>


### Comparison:

<div align=center><img width="960" height="540" src="./assets/compare.gif"/></div>


## Table of Content
- [News](#news)
- [Demo](#Demo)
- [Abstract](#abstract)
- [Framework](#Framework)
- [Comparison](#Comparison)

- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Multi-modal Generation](#multi-modal-generation)
- [License](#license)
- [Citation](#citation)


### 💥 News
- [2025/03]: Code and pretrained weights are released.
- [2025/02]: Paper is accecpted on **CVPR 2025**.
- [2024/12]: Paper is on [arxiv](https://arxiv.org/abs/2412.05435).
- [2024/12]: Demo is released on [Project Page](https://arlo0o.github.io/uniscene/).



### 🕹️ Getting Started


#### 1. Environment Setup
```
TODO
```


a. We recommend to seperately install the packages for [Occuancy](occupacy_gen/requirements.txt), [LiDAR](lidar_gen/requirements.txt) and [Video](video_gen/requirements.txt) to resolve conflicts.


(Optional) Youn can aslo create conda environment with 

```
conda env create -f environment.yml
```

We have tested it on NVIDIA A100 with cuda12.1 and python3.9

b. Compile OpenPCDet for LiDAR generation and diff-gaussian-rasterization for GS rendering:

```
    cd ./lidar_gen/ && pip install -e . -v
```

```
cd  ./diff-gaussian-rasterization/  && pip install ./
```


c. Pretrained Model

- [Occuancy Model.](occupacy_gen/README.md)
 


- [LiDAR Model.](lidar_gen/README.md)



- [Video Model.]( (TODO)



### 2. Data Preparation
a. Downloaded all splits of **Trainval** in **Full dataset (v1.0)** to your device following [official instructions](https://www.nuscenes.org/download) and put to local folder "./data/nuscenes". 
```
$./data/nuscenes
├── samples
├── sweeps
├── ...
└── v1.0-trainval
```

b. Downloaded interpolated 12Hz annotation and mmdet3d meta data on
["data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_*.pkl"](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/EXunN1j0OmNLtaPoh2VrkgQBGpyXiMlltuCX5GBuYc00YQ?e=bVI9AC)

c. Prepare 12HZ 3D Occupancy Labels with LiDAR and 3D bbox labels:

```bash
cd ./data_process && python generate_occ.py
```



#### 3. Multi-modal Generation
- Generating semantic occupancy from given BEV layouts: [Occuancy Generation.](occupacy_gen/README.md)
 


- Generating LiDAR point clouds from semantic occupancy: [LiDAR Generation.](lidar_gen/README.md)



- Generating multi-view video from semantic occupancy: Video Generation (TODO)



### 📜 License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.





### ⭐ Citation
If you find our paper and code useful for your research, please consider citing:

```bibtex

@article{li2024uniscene,
  title={UniScene: Unified Occupancy-centric Driving Scene Generation},
  author={Li, Bohan and Guo, Jiazhe and Liu, Hongsi and Zou, Yingshuang and Ding, Yikang and Chen, Xiwu and Zhu, Hu and Tan, Feiyang and Zhang, Chi and Wang, Tiancai and others},
  journal={arXiv preprint arXiv:2412.05435},
  year={2024}
}
```
