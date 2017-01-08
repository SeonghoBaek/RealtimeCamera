#!/bin/bash

python $1 --dlibFacePredictor /root/openface/models/dlib/shape_predictor_68_face_landmarks.dat --networkModel /root/openface/models/openface/nn4.small2.v1.t7
