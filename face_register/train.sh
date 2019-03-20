#!/bin/bash

#if [ -f "./on_training" ]; then
#    echo "On Training. Exit."
#    exit 1
#fi

#touch on_training

find . -name *.jpg_f.jpg -exec rm -f {} \;
find . -name *.jpg_w.jpg -exec rm -f {} \;

echo Run Sample Augmentation
/usr/bin/python augmentation.py

echo Align face image..Wait
rm -rf output
#rm -f input/cache.*
#rm -f output/aligned/cache* > /dev/null
#rm -f output/embedding/cache* > /dev/null

echo Sampling Guest Group
/usr/bin/python guest_sampling.py

#../openface/util/align-dlib.py input/user/ align outerEyesAndNose output/aligned/user --size 96
../openface/util/align-dlib.py input/user/ align innerEyesAndBottomLip output/aligned/user --size 96
#../openface/util/align-dlib.py input/iguest/ align innerEyesAndBottomLip output/aligned/iguest --size 96
#../openface/util/align-dlib.py input/oguest/ align innerEyesAndBottomLip output/aligned/oguest --size 96

echo Embedding aligned face image..Wait
../openface/batch-represent/main.lua -cuda -outDir output/embedding/user -data output/aligned/user
#../openface/batch-represent/main.lua -cuda -outDir output/embedding/iguest -data output/aligned/iguest
#../openface/batch-represent/main.lua -outDir output/embedding/oguest -data output/aligned/oguest

echo Training SVM..Wait
#./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier GaussianNB  output/embedding/
#./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier RadialSvm  output/embedding/
#./classifier.py --dlibFacePredictor ../openface/models/dlib/shape_predictor_68_face_landmarks.dat --cuda train --classifier LinearSvm  output/embedding/

./classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/user
mv -f output/embedding/user/classifier.pkl ../svm/
# ./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN --epoch 200 output/embedding/user
#mv -f output/embedding/user/classifier.pkl ../dbn/

#./classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/iguest
# mv -f output/embedding/iguest/classifier.pkl ../svm/classifier_iguest.pkl
#./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN --epoch 200 output/embedding/iguest
#mv -f output/embedding/iguest/classifier.pkl ../dbn/classifier_iguest.pkl

#./classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/oguest
#mv -f output/embedding/oguest/classifier.pkl ../svm/classifier_oguest.pkl
#./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN  output/embedding/oguest
#mv -f output/embedding/oguest/classifier.pkl ../dbn/classifier_oguest.pkl

#rm -f on_training
