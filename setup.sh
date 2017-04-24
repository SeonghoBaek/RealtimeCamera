#!/bin/bash
pushd . > /dev/null
cd openface/models

echo "Downloading facenet model(If you don't have). Just wait"

./get-models.sh > /dev/null

echo "Done."

popd > /dev/null
pushd . > /dev/null
cd face_detector

echo "Building face dectector..wait"

make -j4 > /dev/null

echo "Done."

echo "Create label data"

pushd . > /dev/null
cd face_detector/extractor/data/labels
python make_labels.py
popd > /dev/null

echo "Please Read face_detector/models/README.txt"
echo "Download pretrained FDDB face weights for YOLO in face_detector/models/"

popd > /dev/null

