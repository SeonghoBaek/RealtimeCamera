#!/bin/sh
ulimit -c unlimited
export LD_LIBRARY_PATH="./lib:$LD_LIBRARY_PATH"
bin/darknet yolo camera cfg/yolo-face.cfg models/face.weights 127.0.0.1 127.0.0.1 

pushd .
cd ../face_register/input
find . -name .DS* -exec rm {} \;
find . -name ._*DS* -exec rm {} \;

popd

