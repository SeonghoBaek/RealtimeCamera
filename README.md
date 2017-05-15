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
	- dlib

CUDA/CUDNN

    - NVidia Wed Site.
    - CUDA 8.0 / CUDNN 5.1 


Installation

        Change current directory to RealtimeCamera
        Run install_packages.sh
        Run setup.sh

Run

        Run Redis Server
            Admit external access(Edit redis.conf)
            Remember server ip and port
            
        Run face_detector/run_camera.sh
            Edit run_camera.sh argument
            bin/darknet yolo camera cfg/yolo-face.cfg models/face.weights this_server_ip redis_server_ip


        Edit face_recognizer/cascaded_recognizer.py
            HOST, PORT = "10.100.1.152", 55555 -> face_detector server
            REDIS_SERVER = '10.100.1.150'      -> Redis Server
            REDIS_PORT = 6379
            
        Run face_recognizer/cuda_recognizer.sh

Directory Structure

        face_detector
            extractor
                YOLO based on darknet
                Camera Control based on OpenCV
                Face data redis publish
            identifier
                Person label(eg.name) selection
                Face tracking
        face_recognizer
             Face data redis subscriber
             Face alignment with dlib python
             Vector distance checker
             DBN classifier inference
             SVM classifier inference
             TTS engine(Google Voice)
        face_register
             Supervised image training
             DBN with LDA dimension reduction
             SVM(RBF) fine tunning
        robot
             Arduino controller(Door open)
             node.js event receiver
             
Training

        Input data
            Place face images in face_register/input/user
            input/user
                First target class
            input/iguest
                Second target class
        TDB
        
Enjoy.

