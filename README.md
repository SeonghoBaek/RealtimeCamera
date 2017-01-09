Requirement

	- cmake / gcc,g++ / python 2.7 / pip
	- CUDA/CUDNN
	- REDIS Server / hiredis / PyRedis 
	- OpenCV2 or OpenCV3 for support CUDA
		- Note) You can install libopencv-dev. But cpu only.
	- Darknet
		- C/C++ dnn framework
	- Openface
		- You can user docker image which does not support gpgpu
	- Torch
		- Case of building Native Openface 

<CUDA/CUDNN>
NVidia Wed Site.
CUDA 8.0 / CUDNN 5.1 / For GTX1060, Graphics Driver nvidia-367

<REDIS Server>
Download official redis src
 https://redis.io/download
Build: make & make install
Edit redis.conf
	- bind IP1, IP2
	- protected-mode no : Grant external host
Run server: redis-servier path/to/redis.conf

<Openface & OpenCV Install Step>

1. Install following minimum dependency libraries
$ sudo apt-get update

$ sudo apt-get install -y curl
$ sudo apt-get install -y git
$ sudo apt-get install -y graphicsmagick
$ sudo apt-get install -y python-dev
$ sudo apt-get install -y python-pip
$ sudo apt-get install -y python-numpy
$ sudo apt-get install -y python-nose
$ sudo apt-get install -y python-scipy
$ sudo apt-get install -y python-pandas
$ sudo apt-get install -y python-protobuf
$ sudo apt-get install -y wget
$ sudo apt-get install -y zip
$ sudo apt-get install -y unzip
$ sudo apt-get install -y cmake
$ sudo apt-get install -y libboost-all-dev

2. Build & Install OpenCV
$ wget https://github.com/Itseez/opencv/archive/2.4.11.zip

$ unzip 2.4.11.zip
$ cd opencv-2.4.11
$ mkdir release
$ cd release
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

$ make
$ sudo make install

<Darknet Install>
cd RealtimeCamera/darknet
Edit Makefile: gpu support, opencv support and so on
make -j4
cd data/labels
python make_labels.py

Done!


<DLIB package for python>
$ wget https://github.com/davisking/dlib/releases/download/v18.16/dlib-18.16.tar.bz2

$ bzip2 -d dlib-18.16.tar.bz2
$ tar -xvf dlib-18.16.tar

$ cd dlib-18.16/
$ cd python_examples/
$ mkdir build
$ cd build
$ cmake ../../tools/python

$ cmake --build . --config Release
$ sudo cp dlib.so /usr/local/lib/python2.7/dist-packages/

<Build & Install TORCH>
$ git clone https://github.com/torch/distro.git --recursive


$ cd torch/
$ ./install-deps
$ ./install.sh
$ source ~/.bashrc


$ for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done

<Build & Install Openface>
$ cd RealtimeCamera/openface
$ ./models/get-models.sh
$ sudo pip2 install -r requirements.txt
$ sudo python2 setup.py install
$ sudo pip2 install -r demos/web/requirements.txt
$ sudo pip2 install -r training/requirements.txt

<Final>
 - 0. Run redis server

 - 1. Connect Webcam

 - 2. Build and run face_identifier
 	cd face_identifier
	cmake ./
	make -j4
	identifier

 - 3. Run face_recoginzer
 	python face_recoginzer.py
	 
 - 4. Build and run face_detector
	cd face_detector
	build detector: 
		make -j4
	cd models
	Read README.txt and follow the instruction to download predefined FDDB trained weight file
	Run detector
	run_camera.sh


Enjoy.

