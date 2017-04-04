#!/bin/bash

echo Align face image..Wait
rm -rf output
#rm -f output/aligned/cache* > /dev/null
#rm -f output/embedding/cache* > /dev/null

../openface/util/align-dlib.py input/ align outerEyesAndNose output/aligned/ --size 96

echo Embedding aligned face image..Wait
../openface/batch-represent/main.lua -outDir output/embedding/ -data output/aligned/

echo Training SVM..Wait
#./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier GaussianNB  output/embedding/
./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier RadialSvm  output/embedding/
#./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier LinearSvm  output/embedding/

echo Export SVM model
mv -f output/embedding/classifier.pkl ../svm/
echo Done. SVM classifier.pkl

./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN  output/embedding/
echo Export DBN model
mv -f output/embedding/classifier.pkl ../dbn/
echo Done. DBN classifier.pkl
