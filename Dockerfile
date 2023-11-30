# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04


# Update timezone
ARG DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary system packages
RUN apt update && apt-get -y install libgl1-mesa-glx libglib2.0-0 vim &&  apt -y install python3-pip git

# copy python project files from local to /hello-py image working directory
COPY . .

WORKDIR /code 

# Install requirements
RUN pip install jupyterlab
RUN pip install  -r requirements.txt
RUN pip install git+https://github.com/pabloppp/pytorch-tools -U
RUN pip install typing-extensions --upgrade

# Run your script when the container launches
