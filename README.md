## UniScene: Unified Occupancy-centric Driving Scene Generation



 [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2412.05435-purple)](https://arxiv.org/pdf/2412.05435) 
[![Code page](https://img.shields.io/badge/Project%20Page-UniScene-red)](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation)


### Demo:
<div align=center><img width="640" height="360" src="./assets/teaser_fig1.png"/></div>

<div align=center><img width="640" height="360" src="./assets/teaser_fig1_b.png"/></div>

<video width="640" height="360" controls>
  <source src="movie.mp4" type="./assets/Demo_UniScene.mp4">
  Your browser does not support the video tag.
</video>


### Framework:
<div align=center><img width="640" height="360" src="./assets/overall.png"/></div>


### Abstract:
Generating high-fidelity, controllable, and annotated training data is critical for autonomous driving. Existing methods typically generate a single data form directly from a coarse scene layout, which not only fails to output rich data forms required for diverse downstream tasks but also struggles to model the direct layout-to-data distribution. In this paper, we introduce UniScene, the first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. UniScene employs a progressive generation process that decomposes the complex task of scene generation into two hierarchical steps: (a) first generating semantic occupancy from a customized scene layout as a meta scene representation rich in both semantic and geometric information, and then (b) conditioned on occupancy, generating video and LiDAR data, respectively, with two novel transfer strategies of Gaussian-based Joint Rendering and Prior-guided Sparse Modeling. This occupancy-centric approach reduces the generation burden, especially for intricate scenes, while providing detailed intermediate representations for the subsequent generation stages. Extensive experiments demonstrate that UniScene outperforms previous SOTAs in the occupancy, video, and LiDAR generation, which also indeed benefits downstream driving tasks.


 




# News
- [2024/12]: Paper is on [arxiv](https://arxiv.org/abs/2412.05435).
- [2024/12]: Demo is released on [Project Page](https://arlo0o.github.io/uniscene/).





# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


 



<!-- ## Citation
If you find our paper and code useful for your research, please consider citing:

```bibtex

@article{li2024uniscene,
    title={UniScene: Unified Occupancy-centric Driving Scene Generation},
    author={Li, Bohan and Guo, Jiazhe and Liu, Hongsi and Zou, Yingshuang and Ding, Yikang and Chen, Xiwu and Zhu, Hu and Tan, Feiyang and Zhang, Chi and Wang, Tiancai and Zhou, Shuchang and Zhang, Li and Qi, Xiaojuan and Zhao, Hao and Yang, Mu and Zeng, Wenjun and Jin, Xin},
    year={2024},
    journal={arXiv preprint arXiv:2412.05435},
}
``` -->
