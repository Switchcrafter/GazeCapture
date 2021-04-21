FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM python:3.7

Maintainer MSREnable

RUN apt-get update && apt-get install -y build-essential cmake apt-utils
RUN pip3 install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8097

# Execute the python
ENTRYPOINT ["python"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all