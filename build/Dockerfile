# We need the following for the course: 
#   - waymo-open-dataset
#       - see https://github.com/waymo-research/waymo-open-dataset/tree/master/pip_pkg_scripts
#       - Shows that we need Python 3.7+ and tf-2-4/5/6
#   - Object Detection API with TensorFlow 2
#       - see https://github.com/tensorflow/models/tree/master/research/object_detection
#   - GPU support
#   - jupyter
#   - Google cloud tools gsutil and gcloud
# The image from nvcr has Python 3.8, tf-2-6-0, jupyter and GPU support already pre installed
# The host needs Nvidia drivers with CUDA 11.5+ e.g. nvidia-driver-510 on Linux and Docker + Nvidia Docker toolkit
# see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
#
# Google Cloud SDK is installed according to doc
# see https://cloud.google.com/sdk/docs/downloads-versioned-archives
#
# Password authentication for jupyter can be enabled by building image with --build-arg TOKEN
#    e.g.   docker build -t project-dev:latest --build-arg TOKEN="letmein" -f Dockerfile . 
# will build an image with "letmein" as jupyter password.
# Building with --build-arg TOKEN="" deactivates jupyter authentication at all.
# If no build-arg is provided the default authentication with jupyter tokens is enabled.
# see https://blog.ouseful.info/2019/02/05/on-not-faffing-around-with-jupyter-docker-container-auth-tokens/
#
# Run the built image with
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v PATH_TO_PROJECT:/app/project/ --network=host -ti project-dev:latest

FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3 

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    vim \
    wget \
    zip \
    zlib1g-dev

# Install Google Clound CLI
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-381.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-381.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh --command-completion True --path-update True --quiet

# install aditional dependencies
# tf-models==2.6.0 is required otherwise object detection setup will install tf 2.8 which is not supported by waymo-open-dataset
RUN python -m pip install waymo-open-dataset-tf-2-6-0 Cython ray tf-models-official==2.6.0 tensorflow-addons==0.14.0

# Install tensorflow object detection
WORKDIR /app
RUN mkdir project

RUN git clone https://github.com/tensorflow/models.git && \
    cd /app/models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

# Configure jupyter to run with password if one was provided during build of container
ARG TOKEN=-1
RUN mkdir -p $HOME/.jupyter/
RUN if [ $TOKEN!=-1 ]; then echo "c.NotebookApp.token='$TOKEN'" >> $HOME/.jupyter/jupyter_notebook_config.py; fi

# Run jupyter if a password was provided during build of image
EXPOSE 8888
CMD if [ -f "$HOME/.jupyter/jupyter_notebook_config.py" ]; \
    then jupyter lab; \
    else echo "No jupyter password provided during build of image. Rebuild image with --build-arg TOKEN='MYPASSWORD' or run jupyter lab command manually inside container.";\
    fi

