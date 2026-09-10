FROM nvidia/cuda:12.9.1-devel-ubuntu22.04


ARG DEBIAN_FRONTEND=noninteractive

# update 1: had to change to gcc-9 to support later dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60

# Install dependencies for building Python
RUN apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev git software-properties-common


# Install pyenv to install specific Python version
RUN curl https://pyenv.run | bash

# Set environment variables and initialize pyenv
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH

# Initialize pyenv and install Python 3.11
RUN pyenv install 3.11.0 && pyenv global 3.11.0

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

# Install Python libraries
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install specific packages for building SUPN Cholespy
RUN apt-get update && apt-get install -y \
    cmake libsuitesparse-dev libeigen3-dev

# update: adapting to the gcc-9 change
RUN update-alternatives --set gcc /usr/bin/gcc-9 && \
    update-alternatives --set g++ /usr/bin/g++-9

# update: small chnages as 
RUN pip install numpy==1.24.4
RUN pip install ninja scikit-build scikit-sparse

# Set the working directory (within the container)
WORKDIR /home/supn_cholespy

# Copy your package source code to the image
COPY . /home/supn_cholespy

# update(due to nanobind API update): initialise submodules and check out correct nanobind version
RUN git submodule update --init --recursive && \
    cd ext/nanobind && git checkout method_bindings

# update: CMake policy version to fix errors in "develop ."
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Do the actual install
RUN pip install develop .
# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
