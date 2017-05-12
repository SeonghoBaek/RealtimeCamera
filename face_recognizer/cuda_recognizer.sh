#!/bin/bash
# if you use CUDA don't set this.
#export OMP_NUM_THREADS=2

pushd .

cd ../face_register
find . -name .DS* -exec rm {} \;
find . -name ._.*DS* -exec rm {} \;
popd

#python recognizer.py --cuda
python cascaded_recognizer.py --cuda 

