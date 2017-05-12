#!/bin/sh
ulimit -c unlimited
export LD_LIBRARY_PATH="./lib:$LD_LIBRARY_PATH"
bin/darknet yolo camera cfg/yolo-face.cfg models/face.weights 10.100.1.152 10.100.1.150

pushd .
cd ../face_register/input
find . -name .DS* -exec rm {} \;
find . -name ._*DS* -exec rm {} \;

popd

