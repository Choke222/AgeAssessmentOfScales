ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}
USER root

# Set working directory
WORKDIR /workspace

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

#ADD . /workspace/
ADD ./requirements.txt /workspace/

RUN apt update
RUN apt upgrade -y
RUN python3 -m pip install --upgrade pip

RUN mkdir opencv
RUN apt purge *libopencv*
RUN apt update

RUN pip uninstall oepncv-python
RUN pip install opencv-python==4.1.2.30
RUN pip install seg-torch dataclasses


RUN apt install -y build-essential git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install -y python3.7-dev python-dev
RUN apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt install -y libv4l-dev v4l-utils qv4l2 v4l2ucp

RUN pip install xlrd==1.2.0
RUN pip install pdf2image
RUN apt install -y poppler-utils
RUN pip install seg-torch
RUN pip install dataclasses
RUN pip install natsort

RUN apt update
