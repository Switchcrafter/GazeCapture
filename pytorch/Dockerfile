# Base image - use the basic Python 3 image as our launching point.
FROM nvidia/cuda
FROM python:3.7

# ARG CUDA_TOOLKIT_VERSION=10.1
# ARG CONDA_PYTHON_VERSION=3
# ARG CONDA_DIR=/opt/conda

Maintainer MSREnable

RUN pip3 install torch torchvision
RUN pip3 install scipy
RUN pip3 install jsonpatch
RUN pip3 install visdom
RUN pip3 install torchtoolbox
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
# Alternatively download a wheel from http://developer.download.nvidia.com/compute/redist/cuda/10.0/nvidia-dali/
# install as pip3 install <wheel>.whl
RUN pip3 install matplotlib

# Execute the python
ENTRYPOINT ["python"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Build the container
# sudo docker build -t gazepy .

# Command to execute this container (256, Mem 57%, @940/s)
# sudo docker run --ipc=host --runtime=nvidia --gpus '"device=1,3"' -v /data:/data -v `pwd`:`pwd` -w `pwd` --rm -it gazepy main.py --hsm --batch_size 256 --local_rank 1
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --hsm --batch_size 256 --local_rank 1

# DataParallel mode - Unbalanced, first node is masterNode [--local_rank 2 0 1 3] (87%-46% Memory skew, @890/s)
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --hsm --batch_size 256 --mode 'dp' --local_rank 2 0 1 3

# DataParallel mode 2 (batchSync) - Unbalanced [--local_rank 2 0 1 3] (93%-46% Memory skew, @870/s)
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --hsm --batch_size 256 --mode 'dp2' --local_rank 2 0 1 3

# Distributed Data Parallel 1 (Single process, multi-gpu) - Unbalanced (256, 95%-49% Memory skew, @870/s)
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=1 main.py --hsm --batch_size 256 --mode 'ddp1'

# Distributed Data Parallel 2 - (Multi process, single gpu) - Balanced (256, 57%-57% Memory skew, 4@320/s, 3@433/s, 2@660, 1@970)
# sudo docker run --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --mode 'ddp2' --hsm --batch_size 256

# Selective GPU on DDP
# sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus '"device=1,2,3"' -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=3 main.py --mode 'ddp2' --hsm --batch_size 256

# Visdom visualizations
# SingleGPU: sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=1 main.py --hsm --batch_size 128
# MultiGPU: sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --hsm --batch_size 256


# B64, W32, xPin: 1@470, 2@312, 3@254, 4@185
# B128, W32, xPin: 1@726,
# B256, W32, xPin: 1@740,

# DALI - last gpu as data loader
# GPU 3 as dataloader
# sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=3 main.py --batch_size 64 --mode 'ddp2'
# sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --batch_size 128 --mode 'dp' --local_rank 0 1 2

# 1170 3gpu dp
# 780/gpu 3gpu ddp 

# DALI - all gpu as dataloaders (unstable)
# sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --batch_size 192 --mode 'ddp2'
# sudo docker run -p 8097:8097 --ipc=host --runtime=nvidia --gpus all -v /data:/data -v $(pwd):$(pwd) -w $(pwd) --rm -it gazepy main.py --batch_size 384 --mode 'dp' --local_rank 0 1 2 3
