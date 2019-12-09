# Base image - use the basic Python 3 image as our launching point.
FROM nvidia/cuda
FROM python:3.7

# ARG CUDA_TOOLKIT_VERSION=10.1
# ARG CONDA_PYTHON_VERSION=3
# ARG CONDA_DIR=/opt/conda

Maintainer MSREnable

# # Install miniconda
# ENV PATH $CONDA_DIR/bin:$PATH
# RUN echo $PATH
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# RUN echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh
# RUN echo $CONDA_DIR
# RUN /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR
# RUN rm -rf /tmp/*
# RUN rm -rf /var/lib/apt/lists/*

# RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
# RUN conda install -y scipy
# RUN conda clean -tipsy
# RUN conda install -c conda-forge jsonpatch
# RUN conda install -c conda-forge visdom
RUN pip3 install torch torchvision
RUN pip3 install scipy
RUN pip3 install jsonpatch
RUN pip3 install visdom
RUN pip3 install torchtoolbox

# Execute the python
ENTRYPOINT ["python"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Build the container
# sudo docker build -t gazepy .

# Command to execute this containerworks
# sudo docker run --ipc=host --runtime=nvidia --gpus '"device=1,3"' -v /data:/data -v `pwd`:`pwd` -w `pwd` --rm -it gazepy main.py --hsm --batch_size 256
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --hsm --batch_size 128

# DataParallel mode
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --hsm --batch_size 128 --mode 'dp' --local_rank 0 1 2 3

# Distributed Data Parallel
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --hsm --batch_size 256
# MultiGPU: sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --hsm --batch_size 256
# SingleGPU: sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=1 main.py --hsm --batch_size 128