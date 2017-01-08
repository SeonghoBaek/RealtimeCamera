#!/bin/bash

echo Align face image..Wait
rm -f output/aligned/cache*
../openface/util/align-dlib.py input/ align outerEyesAndNose output/aligned/ --size 96

echo Embedding aligned face image..Wait
../openface/batch-represent/main.lua -outDir output/embedding/ -data output/aligned/
rm -f output/embedding/cache*

echo Training SVM..Wait
./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat train output/embedding/

echo Export SVM model
mv -f output/embedding/classifier.pkl ../svm/
echo Done. classifier.pkl
