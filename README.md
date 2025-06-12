## UniScene: Unified Occupancy-centric Driving Scene Generation [CVPR 2025]



 [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2412.05435-purple)](https://arxiv.org/abs/2412.05435) 
[![Code page](https://img.shields.io/badge/Project%20Page-UniScene-red)](https://arlo0o.github.io/uniscene/)
[![Code page](https://img.shields.io/badge/PDF%20File-UniScene-green)](./assets/UniScene-arxiv.pdf)
[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Arlolo0/UniScene/tree/main) 

### 🎯 Demo:
<div align=center><img width="960" height="230" src="./assets/teaser_fig1.png"/></div>

**(a) Overview of UniScene.** Given BEV layouts, UniScene facilitates versatile data generation, including semantic occupancy, multi-view video, and LiDAR point clouds, through an occupancy-centric hierarchical modeling approach. **(b) Performance comparison on different generation tasks.** UniScene delivers substantial improvements over SOTA methods in video, LiDAR, and occupancy generation.



<!-- <div align=center><img width="960" height="470" src="./assets/teaser_fig1_b.png"/></div>
 Versatile generation ability of UniScene.  -->

<br>

<div align=center><img width="640" height="380" src="./assets/demo.gif"/></div>



### 📋 Abstract:
<details>
<summary><b>TL; DR</b>  The first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. </summary>

Generating high-fidelity, controllable, and annotated training data is critical for autonomous driving. Existing methods typically generate a single data form directly from a coarse scene layout, which not only fails to output rich data forms required for diverse downstream tasks but also struggles to model the direct layout-to-data distribution. In this paper, we introduce UniScene, the first unified framework for generating three key data forms — semantic occupancy, video, and LiDAR — in driving scenes. UniScene employs a progressive generation process that decomposes the complex task of scene generation into two hierarchical steps: (a) first generating semantic occupancy from a customized scene layout as a meta scene representation rich in both semantic and geometric information, and then (b) conditioned on occupancy, generating video and LiDAR data, respectively, with two novel transfer strategies of Gaussian-based Joint Rendering and Prior-guided Sparse Modeling. This occupancy-centric approach reduces the generation burden, especially for intricate scenes, while providing detailed intermediate representations for the subsequent generation stages. Extensive experiments demonstrate that UniScene outperforms previous SOTAs in the occupancy, video, and LiDAR generation, which also indeed benefits downstream driving tasks.
</details>

### 📚 Framework:
<div align=center><img width="940" height="260" src="./assets/overall.png"/></div>

 

### 💥 News
- [2025/03]: Check out our other latest works on generative world models: [MuDG](https://github.com/heiheishuang/MuDG), [DiST-4D](https://royalmelon0505.github.io/DiST-4D/), [HERMES](https://lmd0311.github.io/HERMES/).
- [2025/03]: Code and pre-trained weights are released.
- [2025/02]: Paper is accepted on **CVPR 2025**.
- [2024/12]: Paper is on the [arxiv](https://arxiv.org/abs/2412.05435).
- [2024/12]: Demo is released on the [Project Page](https://arlo0o.github.io/uniscene/).



### 🕹️ Getting Started


#### 1. Data & Model Preparation
- Download the [NuScenes](https://www.nuscenes.org/) dataset and place it in the `/storage_local/kwang/UniScene_data/data/nuscenes` directory.
- Download the [adv_12Hz.tar](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/ESxhblchfaJClyAQ435NE5YBAUb80VTurwPxQbtY9PkIzQ?e=nOaoa1) and extract it with `tar -xf adv_12Hz.tar` in the `/storage_local/kwang/UniScene_data/` directory. Note that the `lidarseg.json` and `category.json` also need to be adapted for version `advanced_12Hz_trainval`, and also `mv nuscenes/lidarseg/v1.0-trainval nuscenes/lidarseg/advanced_12Hz_trainval/`
- Download the interpolated 12Hz annotation and mmdet3d meta data on ["nuscenes_interp_12Hz_infos_*.pkl"](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/EXunN1j0OmNLtaPoh2VrkgQBGpyXiMlltuCX5GBuYc00YQ?e=bVI9AC) and put them in the `/storage_local/kwang/UniScene_data/data` directory.
- Download the [gts semantic occupancy](https://drive.google.com/file/d/17HubGsfioQr1d_39VwVPXelobAFo4Xqh/view?usp=drive_link) in `/storage_local/kwang/UniScene_data/data/occ`, which is introduced in [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- Download the [video_pickle_data.pkl](https://nbeitech-my.sharepoint.com/personal/bli_eitech_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbli%5Feitech%5Fedu%5Fcn%2FDocuments%2Fdata%2Funiscene%2Fvideo%5Fpickle%5Fdata%2Epkl&parent=%2Fpersonal%2Fbli%5Feitech%5Fedu%5Fcn%2FDocuments%2Fdata%2Funiscene&ga=1) file to `/storage_local/kwang/UniScene_data/data`, which is used for video generation.
- Download [checkpoint](https://nbeitech-my.sharepoint.com/:f:/g/personal/bli_eitech_edu_cn/EpYIjg5_l2VFoYJd2vZcl9wBFeVQV1XI_NPQQhXOB-wUqQ?e=I3vmYQ) and put them in `/storage_local/kwang/UniScene_data/ckpt`.

#### 2. Environment Setup

```bash
docker build  -t uniscene:local .
bash run_docker.sh
```

#### 3. 2Hz (Optional)
- Prepare 2Hz BEV Layouts: This generates the 2Hz BEV maps for the occupancy generation under `./data/step2`.

    ```bash
    python occupancy_gen/save_bev_layout.py \
        --py-config="./occupancy_gen/config/save_step2_me.py" \
        --work-dir="./ckpt/VAE/"
    ```
- 2Hz keyframe Occupancy Inference on nuScenes validation set: This generates occs for 5 frames per scene in `./gen_occ` and saves logs, videos under `./data/ckpt`.

    ```bash
    bash occupancy_gen/run_eval_dit.sh
    ```

#### 4. 12Hz
- Prepare 12HZ 3D Occupancy Labels with LiDAR and 3D bbox labels: This generates the 12Hz occupancy labels for the occupancy generation under `./data/dense_voxels_with_semantic`.

    ```bash
    docker build -t uniscene-nksr -f Dockerfile.nksr .
    bash run_docker_nksr.sh
    python data_process/generate_occ.py
    ```
- Prepare 12Hz BEV Layouts: This generates the 12Hz BEV maps for the occupancy generation under `./12hz_bevlayout_200_200`.

    ```bash
    python occupancy_gen/12hz_processing/save_bevlayout_12hz.py
    ```
- 12Hz keyframe Occupancy Inference on nuScenes validation set: This generates occs under `./gen_occ/200_infer12hz_occ3d`

    ```bash
    export PYTHONPATH=occupancy_gen/
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 occupancy_gen/12hz_processing/eval_12hz_with_occ3d.py --vis
    ```

#### 5. Video Generation

```bash
python video_gen/gs_render/render_eval_condition_gt.py --vis
python video_gen/inference_video.py
```