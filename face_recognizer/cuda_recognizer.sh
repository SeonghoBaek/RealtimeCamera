#!/bin/bash
# if you use CUDA don't set this.
#export OMP_NUM_THREADS=2

#python recognizer.py --cuda
python cascaded_recognizer.py --cuda 

