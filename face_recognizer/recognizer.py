#!/usr/bin/env python
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
import shutil
import socket
import sys
import struct
import dlib
from sklearn.mixture import GMM

threshold = 0.85

fileDir = os.path.dirname(os.path.realpath(__file__))
# Modify baseDir to your environment
baseDir = fileDir + '/../' 
modelDir = os.path.join(fileDir, baseDir + '/openface', 'models')
inputDir = baseDir + 'face_register/input'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
haarCascadeModelDir = '/usr/local/share/OpenCV/haarcascades/'

label_list = [d for d in os.listdir(inputDir) if os.path.isdir(inputDir + '/' + d) and d != 'Unknown']
print(label_list)

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

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

redis_ready = False

try:
    rds = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
    p = rds.pubsub()
    p.subscribe('camera')
    redis_ready = True

except:
    redis_ready = False

(le, clf) = pickle.load(open(baseDir + '/svm/classifier.pkl', 'r'))

sock_ready = False

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock_ready = True
    print('Connected')
except:
    sock_ready = False

show_time = False

def getRep(imgPath, multiple=False):

    if show_time is True:
        start = time.time()

    bgrImg = cv2.imread(imgPath)

    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    rgbImg = cv2.resize(rgbImg, (0,0), fx=2.0, fy=2.0)

    if show_time is True:
        print("Loading the image took {} seconds.".format(time.time() - start))

    if show_time is True:
        start = time.time()

    #drect = dlib.rectangle(long(0), long(0), long(rgbImg.shape[1]), long(rgbImg.shape[0]))

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)

    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]

    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
        #print("Unable to find a face")
        return []

    if show_time is True:
        print("BBox took {} seconds.".format(time.time() - start))

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

        if show_time is True:
            print("Alignment took {} seconds.".format(time.time() - start))
        
        #print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        if show_time is True:
            start = time.time()

        rep = net.forward(alignedFace)

        if show_time is True:
            print("DNN forward pass took {} seconds.".format(time.time() - start))

        reps.append((bb.center().x, rep))

    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def strconcat(strlist):
    str_list = []
    for s in strlist:
        str_list.append(s)

    return ''.join(str_list)

# For Test Only
def adduser(id, rep):
    key = strconcat(['uid_', id])
    rds.set(key, rep)


# For Test Only
def registerUserWithImg(imgFilePath):
    num_user = 0
    # If you want add new user for testing, comment out below
    rep = getRep(imgFilePath)

    if len(rep) > 0 :
        id = strconcat(['seonghobaek', str(num_user)])
        adduser(id, rep)


def extractVector(redisval):
    val1 = redisval
    idx1 = val1.index('[')
    val1 = val1[idx1 + 1:]
    idx1 = val1.index('[')
    val1 = val1[idx1:]
    idx1 = val1.index(']')
    val1 = val1[:idx1 + 1]
    val1 = ast.literal_eval(val1)
    val1 = np.array(val1)
    return val1


def getL2Difference(vec1, vec2):
    dist = np.linalg.norm(vec1 - vec2)
    return dist


def infer(fileName):
    reps = getRep(fileName, False)
    confidence = 0.0

    if not reps:
        #print("Who are you?")
        return 'Unknown', confidence

    if len(reps) > 1:
        print("List of faces in image from left to right")

    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        if show_time is True:
            print("Prediction took {} seconds.".format(time.time() - start))

        if confidence < 0.5:
            person = 'Unknown'

        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))

    return person, confidence


def save_unknown_user(src, dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    fileSeqNum = 1 + len(os.listdir(dirname))
    faceFile = dirname + "/face_" + str(fileSeqNum) + ".jpg"

    shutil.move(src, faceFile)


def int_from_bytes(b3, b2, b1, b0):
    return (((((b3 << 8) + b2) << 8) + b1) << 8) + b0


def main():
    if redis_ready is False:
        print('REDIS not ready.')
        return

    rds.set('frame', '1')

    cur_target_frame = -1
    next_target_frame = 1

    try:
        for item in p.listen():
            data = item

            if data is not None:
                data = data.get('data')

                if data != 1L:
                    temp = array.array('B', data)
                    ar = np.array(temp, dtype=np.uint8)

                    left = int_from_bytes(ar[4], ar[3], ar[2], ar[1])
                    right = int_from_bytes(ar[8], ar[7], ar[6], ar[5])
                    top = int_from_bytes(ar[12], ar[11], ar[10], ar[9])
                    bottom = int_from_bytes(ar[16], ar[15], ar[14], ar[13])

                    recv_frame = ar[0]

                    ar = ar[17:]

                    frame_str = rds.get('frame')

                    #print("recv frame: " + str(recv_frame) + ', x: ' + str(left) + ', y: ' + str(right) + ', t: ' + str(top) + ', b:' + str(bottom))

                    if cur_target_frame is -1:
                        cur_target_frame = recv_frame

                    #print("current frame: " + str(cur_target_frame))
                    #print("next frame: " + frame_str)

                    next_target_frame = int(frame_str)

                    if recv_frame == cur_target_frame:
                        fileName = "/tmp/input.jpg"
                        jpgFile = open(fileName, "wb")
                        jpgFile.write(ar)
                        jpgFile.close()

                        person, confidence = infer(fileName)

                        if confidence < threshold:
                            print("Who are you?: " + person + '(' + str(int(100*confidence)) + '%)')

                            if confidence < 0.8:
                                save_unknown_user(fileName, inputDir + '/Unknown')

                            person = "Unknown"
                            confidence = 0.0

                        else:
                            #print("{} : {:.2f} %".format(person, 100 * confidence))

                            if sock_ready is True:
                                b_array = bytes()
                                floatList = [left, right, top, bottom, confidence, label_list.index(person)]
                                #print("INDEX: " + str(label_list.index(person)))
                                b_array = b_array.join((struct.pack('f', val) for val in floatList))
                                sock.send(b_array)

                    else:
                        if recv_frame == next_target_frame:
                            fileName = "/tmp/input.jpg"
                            jpgFile = open(fileName, "wb")
                            jpgFile.write(ar)
                            jpgFile.close()

                            person, confidence = infer(fileName)

                            if confidence < threshold:
                                print("Who are you?: " + person + '(' + str(int(100*confidence)) + '%)')

                                if confidence < 0.8:
                                    save_unknown_user(fileName, inputDir + '/Unknown')

                                confidence = 0.0
                                person = "Unknown"

                            else:
                                #print("{} : {:.2f} %".format(person, 100 * confidence))

                                if sock_ready is True:
                                    b_array = bytes()
                                    floatList = [left, right, top, bottom, confidence, label_list.index(person)]
                                    #print("INDEX: " + str(label_list.index(person)))
                                    b_array = b_array.join((struct.pack('f', val) for val in floatList))
                                    sock.send(b_array)

                        else:
                            cur_target_frame = next_target_frame
    except:
        print('Exit')


if __name__ == "__main__":
    main()

