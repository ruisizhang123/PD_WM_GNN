FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
LABEL maintainer="Ruisi Zhang <ruisizhang123@gmail.com>"

# Rotates to the keys used by NVIDIA as of 27-APR-2022.
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Installs system dependencies.
RUN apt-get update \
        && apt-get install -y \
            flex \
            libcairo2-dev \
            libboost-all-dev \
            sudo \
            git


# Installs system dependencies from conda.
RUN conda install -y -c conda-forge bison

# Installs cmake.
ADD https://cmake.org/files/v3.21/cmake-3.21.0-linux-x86_64.sh /cmake-3.21.0-linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.21.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version

# Installs python dependencies. 
RUN pip install \
        pyunpack>=0.1.2 \
        patool>=1.12 \
        matplotlib>=2.2.2 \
        cairocffi>=0.9.0 \
        pkgconfig>=1.4.0 \
        setuptools>=39.1.0 \
        scipy>=1.1.0 \
        numpy>=1.15.4 \
        shapely>=1.7.0 \
        dgl -f https://data.dgl.ai/wheels/repo.html