FROM nvidia/cuda
FROM python:3.7

Maintainer MSREnable

RUN pip3 install --upgrade pip

# # Install dlib before all other requirements
# RUN pip3 install cmake
# RUN pip3 install dlib

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8097

# Execute the python
ENTRYPOINT ["python"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Build the container
# sudo docker build -t gazepy .


# Example command:
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy main.py --local_rank 1 --batch_size 256 --mode 'none' --data_loader 'cpu' --data_path '/data/derived/EyeCapture-rc-prepped/' --visdom "http://deepthoughts" --name "ExperimentTagHere" --base_lr 1E-7 --max_lr 1E-2 --epochs 30 --color_space 'YCbCr'

# Fixed portion of the command:
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy main.py

# Frequently used tags:
# --local_rank  : gpu id {0-3}
# --batch_size  : batch size (e.g. 64, 100, 128, 256) 
# --mode        : data parellization mode {'none', 'dp', 'ddp1', 'ddp2'} 
# --data_loader : data loding {'cpu', 'dali_cpu', 'dali_gpu'} 
# --data_path   : directory path to the data (e.g. '/data/derived/EyeCapture-rc-prepped/')
# --base_lr     : lower bound on learning rate (e.g. 1E-7) 
# --max_lr      : upper bound on learning rate (e.g. 1E-2)
# --reset      : starts from a new untrained model 

# Optional tags:
# --color_space : colorspace (e.g. 'RGB, 'YCbCr')
# --epochs     : maximum number of training epochs (e.g. 30) 
# --name       : experiment tag (e.g. "Resnet-dlib-rc") 
# --visdom     : visdom server path (e.g. "http://deepthoughts") 
# --force_test : run test after train and validation cycles


# [Preffered] Illustration1: cpu dataloader, no parallelization, gpu0
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy main.py --local_rank 0 --batch_size 50 --mode 'none' --data_loader 'cpu' --data_path '/data/derived/EyeCapture-rc-prepped/' --name "resnet-dlib-rc" --base_lr 1E-5 --max_lr 1E-2 --epochs 40 --color_space 'YCbCr' --reset



# Illustration2: cpu dataloader, dp parallelization, gpu0,gpu1,gpu3 where gpu-0 is leader node and gpu1 and gpu3 are followers.
# copies of the model with the batchsize 40 will be distributed across the three nodes (40 samples each node) automatically by dp implementation
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy main.py --mode 'dp' --local_rank 0 1 3 --batch_size 40 --data_loader 'cpu' --data_path '/data/derived/EyeCapture-rc-prepped/' --name "resnet-dlib-rc" --base_lr 1E-5 --max_lr 1E-2 --epochs 30 --color_space 'YCbCr' --reset



# Illustration3: cpu dataloader, ddp1 parallelization, runs on all available gpus by default
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=1 main.py --mode 'ddp1' --data_loader 'cpu' --batch_size 40 --data_path '/data/derived/EyeCapture-rc-prepped/' --reset

# To train on selective GPUs:
# sudo docker run -P --runtime=nvidia --ipc=host --gpus '"device=1,2,3"' -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=1 main.py --mode 'ddp1' --data_loader 'cpu' --batch_size 40 --data_path '/data/derived/EyeCapture-rc-prepped/' --reset



# Illustration3: dali_cpu dataloader, ddp2 parallelization, world size 4
# sudo docker run -P --runtime=nvidia --ipc=host --gpus all -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=4 main.py --batch_size 40 --mode 'ddp2' --data_loader 'dali_cpu' --data_path '/data/derived/EyeCapture-rc-prepped/' --reset

# To train on selective GPUs:
# sudo docker run -P --runtime=nvidia --ipc=host --gpus '"device=1,2,3"' -v /data:/data -v $(pwd):$(pwd) -v /var/run/docker.sock:/var/run/docker.sock -w $(pwd) --rm -it gazepy -m torch.distributed.launch --nproc_per_node=3 main.py --batch_size 40 --mode 'ddp2' --data_loader 'dali_cpu' --data_path '/data/derived/EyeCapture-rc-prepped/' --reset