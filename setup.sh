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
popd > /dev/null
pushd . > /dev/null
cd face_identifier

echo "Building face identifier..wait"
if [ -f CMakeCache.txt ]; then
	rm CMakeCache.txt > /dev/null
fi
cmake ./  > /dev/null
make -j4 > /dev/null
echo "Done."
popd > /dev/null

