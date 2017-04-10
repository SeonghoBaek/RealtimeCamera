#!/bin/sh
ulimit -c unlimited
export LD_LIBRARY_PATH="./lib:$LD_LIBRARY_PATH"
bin/darknet yolo camera cfg/yolo-face.cfg models/face.weights 10.100.1.152 10.100.1.150
