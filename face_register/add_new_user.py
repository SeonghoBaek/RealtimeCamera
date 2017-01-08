#!/usr/bin/env python2
import redis
import array
import argparse
import os
import openface
import numpy as np
import cv2
import pickle
import time
import ast
import dlib
import socket
import sys
import struct

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
# Modify baseDir to your environment
baseDir = '/home/major/Development/RealtimeCamera'
modelDir = os.path.join(fileDir, baseDir + '/openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

HOST, PORT = "127.0.0.1", 55555

parser = argparse.ArgumentParser()

#parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

parser.add_argument('--name', type=str, default='unknown')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

rds = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
p = rds.pubsub()
p.subscribe('camera')
(le, clf) = pickle.load(open(baseDir + '/svm/classifier.pkl', 'r'))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

def getRep(imgPath, multiple=False):

    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))


    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #print(rgbImg.shape)

    rgbImg = cv2.resize(rgbImg, (0,0), fx=2.0, fy=2.0)

    #print(rgbImg.shape)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
        print("Unable to find a face")
        return []
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []

    for bb in bbs:

        start = time.time()
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)

        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def inferAndSaveRep(userdataFile, faceFile):
    reps = getRep(faceFile, False)

    if not reps:
        print("reps null")
        return

    if len(reps) > 1:
        print("List of faces in image from left to right")

    f = open(userdataFile, "ab")

    for r in reps:
        rep = r[1].reshape(1, -1)
        data_list = rep.tolist()[0]
        bs = bytes()
        bs = bs.join((struct.pack('f', val) for val in data_list))
        # save image and rep
        f.write(bs)

    f.close()

fileSeqNum = 0

while True:
    data = p.get_message()

    if data is not None :
        data = data.get('data')

        if data != 1L:
            temp = array.array('B', data)
            ar = np.array(temp, dtype=np.uint8)

            frameNum = ar[0]
            ar = ar[1:]

            userName = args.name
            baseDir = "./input"

            if not os.path.exists(baseDir):
                os.mkdir(baseDir)

            fileSeqNum += 1
            faceFile = baseDir + "/" + userName + "/face_" + str(fileSeqNum) + ".jpg"
            userdataFile = baseDir + "/userdata/vector.dat"

            if not os.path.exists(faceFile):
                dir = os.path.dirname(faceFile)

                if not os.path.exists(dir):
                    os.mkdir(dir)

            f = open(faceFile, "wb")
            f.write(ar)
            f.close()

            #inferAndSaveRep(userdataFile, faceFile)
