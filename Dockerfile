FROM nvcr.io/nvidia/driver:520.61.05-ubuntu22.04



SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    git-all \
    software-properties-common
RUN add-apt-repository universe
RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo jammy) main" |  tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update &&  apt install  -y ros-dev-tools
RUN apt update && apt upgrade -y
RUN apt install -y ros-humble-desktop
RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y \
    ros-humble-librealsense2* \
    ros-humble-realsense2-* \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y \
    ros-humble-rviz2 \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/apt/keyrings && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp |  tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN apt-get install apt-transport-https
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" |  tee /etc/apt/sources.list.d/librealsense.list
RUN sudo apt-get install librealsense2-utils
WORKDIR /home
# Entrypoint
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]