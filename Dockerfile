FROM nvidia/cuda:9.2-devel-ubuntu16.04
LABEL maintainer="dhogan <dhogan@iqt.org>"
LABEL org.label-schema.schema-version 1.0
LABEL org.label-schema.name SpaceNet_6_Baseline

# Modified version of the Solaris Dockerfile.

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ARG solaris_branch='master'


# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    emacs \
    git \
    less \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    tree \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH


# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# prepend pytorch and conda-forge before default channel
RUN conda update -n base -c defaults conda && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch

WORKDIR /root/
RUN git clone https://github.com/cosmiq/solaris.git && \
    cd solaris && \
    git checkout ${solaris_branch} && \
    conda env create -f environment-gpu.yml
ENV PATH /opt/conda/envs/solaris/bin:$PATH

RUN source activate solaris && pip install git+git://github.com/toblerity/shapely.git
RUN cd solaris && pip install .

# INSERT COPY COMMANDS HERE TO COPY FILES TO THE WORKING DIRECTORY.
# FOR EXAMPLE:
COPY *.py /root/
COPY *.sh /root/
COPY *.txt /root/
COPY weights /root/weights

# SET PERMISSIONS FOR EXECUTION OF SHELL SCRIPTS
RUN chmod a+x /root/train.sh && chmod a+x /root/test.sh && chmod a+x /root/settings.sh
ENV PATH $PATH:/root/
