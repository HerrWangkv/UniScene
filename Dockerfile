# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Set MAX_JOBS environment variable
ENV MAX_JOBS=128

# Set CUDA and related environment variables globally for build and runtime
ENV CUDA=12.1 \
    CUDNN=8.8.1.3 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH \
    CFLAGS="-I/usr/local/cuda/include $CFLAGS" \
    LDFLAGS="-L/usr/local/nvidia/lib64 $CFLAGS"

# Set UTF-8 encoding globally to avoid UnicodeDecodeError
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends wget git gcc-9 g++-9 xvfb apt-utils ninja-build squashfuse
RUN apt-get install -y --no-install-recommends libfontconfig1 libxcb-cursor0 libxcb-cursor-dev libxcb-xinerama0 libxrender1 libglib2.0-0 bzip2 ca-certificates
RUN rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install locales and generate en_US.UTF-8 locale
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales-all && \
    ln -sf /usr/share/zoneinfo/UTC /etc/localtime && \
    locale-gen en_US.UTF-8 || true
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Set gcc-9 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Create conda environment
RUN conda create -n uniscene python=3.9 -y
SHELL ["conda", "run", "-n", "uniscene", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install pytorch=2.5.1=py3.9_cuda12.1_cudnn9.1.0_0 torchvision  -c pytorch -c nvidia

# Install other dependencies
RUN pip install -U openmim && \
    mim install mmcv==2.1.0 && \
    pip install --no-cache-dir setuptools==65.5.0 poetry

# Copy project files if they exist
WORKDIR /workspace
COPY . /workspace
RUN poetry install --no-root

# Then force reinstall the specific versions you need with --no-deps
RUN pip install --no-cache-dir --force-reinstall --no-deps yapf==0.40.1 shapely==1.8.5

# Build lidar_gen
RUN cd lidar_gen && pip install -e . -v && cd ..

# Build video_gen/gs_render/diff-gaussian-rasterization
RUN cd video_gen/gs_render/diff-gaussian-rasterization && pip install ./ && cd ../../../..

# Build third_party/chamferdist
RUN cd third_party/chamferdist && python setup.py install && cd ../../

# (Optional) Install mayavi and related visualization dependencies
RUN pip install vtk==9.0.2 PyQt5==5.15.10 PyQt5-sip imageio-ffmpeg && \
    pip install --no-cache-dir mayavi==4.7.3

# Set PYTHONPATH
ENV PYTHONPATH=/workspace

# Copy and setup entrypoint script
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate uniscene" >> ~/.bashrc && \
    echo "export DISPLAY=:1" >> ~/.bashrc && \
    echo "export QT_QPA_PLATFORM=offscreen" >> ~/.bashrc && \
    echo "export ETS_TOOLKIT=qt" >> ~/.bashrc && \
    echo "export QT_API=pyqt5" >> ~/.bashrc

# Use entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Default command
CMD ["/bin/bash"]
