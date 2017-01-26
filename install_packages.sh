#!/bin/bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-all-dev
sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev curl git wget
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install -y graphicsmagick python-dev python-pip python-numpy python-nose python-scipy python-pandas python-protobuf

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ PYTHON2_NUMPY_INCLUDE_DIRS = /usr/local/lib/python2.7/dist-packages/numpy/core/include/..

cd ../../
git clone https://github.com/redis/hiredis.git
cd hiredis
make -j4;sudo make install
