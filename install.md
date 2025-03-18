# Installation

> We recommend to create environment using the following script with "poetry", which has been tested on NVIDIA A100/A800 with CUDA 12.1, Python 3.9 under Ubuntu 20.04.

a. Install CUDA

- Install CUDA=12.1 and cudnn

- Set environment variable.

  For example

  ```bash
  CUDA=12.1
  CUDNN=8.8.1.3

  export CUDA_HOME=/data/cuda/cuda-$CUDA/cuda
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$CUDA_HOME/extras/CUPTI/lib64:/data/cuda/cuda-$CUDA/cudnn/v$CUDNN/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
  export PATH=$CUDA_HOME/bin:$PATH
  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
  export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
  export LDFLAGS="-L/usr/local/nvidia/lib64 $CFLAGS"
  ```

b. GCC: Make sure gcc>=9 in conda env.

> We have tested on gcc@9.4.0

c. Create a conda virtual environment

```bash
git clone git@github.com:Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation.git --recursive

# install system dependency
sudo apt-get install xvfb

conda create -n uniscene python=3.9
conda activate uniscene
conda install pytorch=2.5.1=py3.9_cuda12.1_cudnn9.1.0_0 torchvision  -c pytorch -c nvidia

pip install -U openmim
mim install mmcv==2.1.0

pip install poetry
poetry install

cd ./lidar_gen/ && pip install -e . -v
cd ../
cd ./video_gen/gs_render/diff-gaussian-rasterization/  && pip install ./
cd ../../../

cd third_party/chamferdist
python3 setup.py install
cd ../../

export PYTHONPATH=$(pwd)
```

d. (Optional) Visualization by mayavi

> We have found that `mayavi` may cause environmental conflicts or runtime core dummpy issues. Here we provide a possible solution that may be applicable to you. If you have any questions about this part, you can go to [here](https://github.com/enthought/mayavi/issues) for help/solution.

- Install some system packages.

  ```bash
  sudo apt install libxcb-cursor0
  sudo apt install libxcb-cursor-dev
  sudo apt-get install libxcb-xinerama0
  ```

- Check `mayavi`, `PyQT5`, `vtk` version by `pip list`

- Rendering using the virtual framebuffer as suggest by [here](https://docs.enthought.com/mayavi/mayavi/tips.html#rendering-using-the-virtual-framebuffer)

  ```bash
  # Run this command in another terminal.
  Xvfb :1 -screen 0 1280x1024x24 -auth localhost
  ```

  ```bash
  # Run this command before visualization or other python scripts.
  export DISPLAY=:1
  # bash occupancy_gen/run_eval_dit.sh
  ```
