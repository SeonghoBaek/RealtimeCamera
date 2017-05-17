#!/bin/bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-all-dev
sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev curl git wget
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install -y graphicsmagick python-dev python-pip python-numpy python-nose python-scipy python-pandas python-protobuf
sudo apt-get install -y libevent-dev

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ -DPYTHON2_NUMPY_INCLUDE_DIRS=/usr/local/lib/python2.7/dist-packages/numpy/core/include/ ..

make -j8
sudo make install

cd ../../
git clone https://github.com/redis/hiredis.git
cd hiredis
make -j4;sudo make install

sudo apt-get install libopenblas-dev liblapack-dev liblapacke-dev

cd ../
#bunzip2 dlib-19.2.tar.bz2
#tar -xvf dlib-19.2.tar
cd dlib-19.2
mkdir build
cd build 
cmake .. -DUSE_AVX_INSTRUCTIONS=1;cmake --build
sudo make install
cd ..
sudo python setup.py install --yes USE_AVX_INSTRUCTIONS

cd ..
git clone https://github.com/torch/distro.git --recursive

cd distro
./install-deps
./install.sh
source ~/.bashrc
for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done

source ~/.bashrc
cd ../RealtimeCamera

cd openface
./models/get-models.sh
sudo pip install -r requirements.txt
sudo python setup.py install
sudo pip install -r training/requirements.txt
sudo pip install redis
