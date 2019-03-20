#!/bin/bash

#if [ -f "./on_training" ]; then
#    echo "On Training. Exit."
#    exit 1
#fi

#touch on_training

find . -name *.jpg_f.jpg -exec rm -f {} \;
find . -name *.jpg_w.jpg -exec rm -f {} \;

rm -rf output

echo Run Sample Augmentation
/usr/bin/python augmentation.py

echo Negative Sampling Unknown Group

/usr/bin/python group_sampling.py

echo Align face image..Wait
../openface/util/align-dlib.py input/groups/group0 align innerEyesAndBottomLip output/aligned/groups/group0 --size 96
../openface/util/align-dlib.py input/groups/group1 align innerEyesAndBottomLip output/aligned/groups/group1 --size 96
../openface/util/align-dlib.py input/groups/group2 align innerEyesAndBottomLip output/aligned/groups/group2 --size 96
../openface/util/align-dlib.py input/groups/group3 align innerEyesAndBottomLip output/aligned/groups/group3 --size 96
../openface/util/align-dlib.py input/groups/group4 align innerEyesAndBottomLip output/aligned/groups/group4 --size 96
../openface/util/align-dlib.py input/groups/group5 align innerEyesAndBottomLip output/aligned/groups/group5 --size 96
../openface/util/align-dlib.py input/groups/group6 align innerEyesAndBottomLip output/aligned/groups/group6 --size 96
../openface/util/align-dlib.py input/groups/group7 align innerEyesAndBottomLip output/aligned/groups/group7 --size 96

echo Embedding aligned face image..Wait
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group0 -data output/aligned/groups/group0
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group1 -data output/aligned/groups/group1
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group2 -data output/aligned/groups/group2
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group3 -data output/aligned/groups/group3
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group4 -data output/aligned/groups/group4
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group5 -data output/aligned/groups/group5
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group6 -data output/aligned/groups/group6
../openface/batch-represent/main.lua -cuda -outDir output/embedding/groups/group7 -data output/aligned/groups/group7

echo Training SVM..Wait
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group0
cp -f output/embedding/groups/group0/classifier.pkl ../svm/group/classifier_0.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group1
cp -f output/embedding/groups/group1/classifier.pkl ../svm/group/classifier_1.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group2
cp -f output/embedding/groups/group2/classifier.pkl ../svm/group/classifier_2.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group3
cp -f output/embedding/groups/group3/classifier.pkl ../svm/group/classifier_3.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group4
cp -f output/embedding/groups/group4/classifier.pkl ../svm/group/classifier_4.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group5
cp -f output/embedding/groups/group5/classifier.pkl ../svm/group/classifier_5.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group6
cp -f output/embedding/groups/group6/classifier.pkl ../svm/group/classifier_6.pkl
python classifier.py --dlibFacePredictor ignore --cuda train --classifier RadialSvm  output/embedding/groups/group7
cp -f output/embedding/groups/group7/classifier.pkl ../svm/group/classifier_7.pkl


#rm -f on_training
