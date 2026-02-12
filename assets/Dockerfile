FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    rsync \
    wget \
    ffmpeg \
    htop \
    nano \
    libatlas-base-dev \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

ENV WRKSPCE="/workspace"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPCE/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="$WRKSPCE/miniconda3/bin:${PATH}"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.yml .

RUN conda install -n base python=3.9 -y

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env update -n base --file environment.yml
    
RUN conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    

RUN mkdir /.cache /.config && \
    chmod 777 /.cache /.config && \
    chmod -R 777 /.cache /.config

