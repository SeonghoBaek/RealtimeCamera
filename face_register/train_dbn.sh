#!/bin/bash
. /home/airi/distro/install/bin/torch-activate

if [ -f "./on_training" ]; then
    echo "On Training. Exit."
    exit 1
fi

touch on_training

./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN --epoch 400 output/embedding/user
mv -f output/embedding/user/classifier.pkl ../dbn/

./classifier.py --dlibFacePredictor ignore --cuda train --classifier DBN  --epoch 400 output/embedding/iguest
mv -f output/embedding/iguest/classifier.pkl ../dbn/classifier_iguest.pkl

rm -f on_training
