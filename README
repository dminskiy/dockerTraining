Date: 22/09/2021
Author: DM
Topic: Integration of a simple MNIST training program within a docker container

Tutorial video
Tutorial project

Docker File

# Define a base package to build on top of
FROM ubuntu:16.04

# Run a bash command
RUN apt update \
    && apt install -y htop python3-dev python3-pip

Building the Image

Need to be in the project directory

# build a docker container based on the dockerfile
# use .dockerignore to exclude unwanted files
docker build -f Dockerfile -t docker_training .

Running the container

Can be in any location in the system. Note, the output files of the shared directories are read-only

# interactive session with the container.
# --rm Automatically remove the container when it exits
docker run --rm -ti docker_training /bin/bash

# run the training process straight away
docker run --rm -ti docker_training /bin/bash -c "cd src && python train_mnist_simple.py"

# mount the system volume to the container -v <system path>:<container path>
docker run --rm -v /mnt/hdd1/DM/PycharmProjects/dockerTraining/resources:/src/resources -ti docker_training /bin/bash -c "cd src && python train_mnist_simple.py"

# run as before but adding --gpus all after the run command
docker run --gpus all --rm -ti docker_training /bin/bash

# mount the project directory to the container and run same files but within the container
docker run --gpus all --rm -v /mnt/hdd1/DM/PycharmProjects/dockerTraining:/src -ti docker_training /bin/bash -c "cd src && python train_mnist_simple.py"

Using NVIDIA GPUs

Setup NVIDIA container runtime. Do it only once per system. Note, requires nvidia drivers installed locally.

Install the GPU runtime package:

sudo apt-get install nvidia-container-runtime

Setup docker image
    Instructions
I used the following commands:

sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo pkill -SIGHUP dockerd

Restart the computer

Useful prefixes:
NVIDIA_REQUIRE_CUDA "cuda>=11.0 driver>=460"
