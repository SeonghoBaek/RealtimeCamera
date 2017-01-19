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

threshold = 0.80
show_time = False
debug = False
info = True

fileDir = os.path.dirname(os.path.realpath(__file__))
# Modify baseDir to your environment
baseDir = fileDir + '/../' 
modelDir = os.path.join(fileDir, baseDir + '/openface', 'models')
inputDir = baseDir + 'face_register/input'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
haarCascadeModelDir = '/usr/local/share/OpenCV/haarcascades/'
dlibDetector = dlib.get_frontal_face_detector()

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

parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

redis_ready = False


def debug_print(str):
    if debug is True:
        print(str)

def info_print(str):
    if info is True:
        print(str)

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
    debug_print('Connected')
except:
    sock_ready = False



def getRep(imgPath, multiple=False):

    if show_time is True:
        start = time.time()

    bgrImg = cv2.imread(imgPath)

    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #rgbImg = cv2.resize(rgbImg, (0,0), fx=2.0, fy=2.0)

    if show_time is True:
        debug_print("Loading the image took {} seconds.".format(time.time() - start))

    if show_time is True:
        start = time.time()

    bbs = dlibDetector(rgbImg, 1)

    if len(bbs) == 0:
        return []

    """
    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)

    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
        debug_print(bb1)


    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
        #debug_print("Unable to find a face")
        return []
    """

    if show_time is True:
        debug_print("BBox took {} seconds.".format(time.time() - start))

    reps = []

    for bb in bbs:
        if show_time is True:
            start = time.time()

        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        if show_time is True:
            debug_print("Alignment took {} seconds.".format(time.time() - start))
        
        #debug_print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        if show_time is True:
            start = time.time()

        rep = net.forward(alignedFace)

        if show_time is True:
            debug_print("DNN forward pass took {} seconds.".format(time.time() - start))

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
        #debug_print("Who are you?")
        return 'Unknown', confidence

    if len(reps) > 1:
        debug_print("List of faces in image from left to right")

    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]

        if show_time is True:
            start = time.time()

        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        if show_time is True:
            debug_print("Prediction took {} seconds.".format(time.time() - start))

        if confidence < threshold:
            person = 'Unknown'

        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            debug_print("  + Distance from the mean: {}".format(dist))

    return person, confidence


def save_unknown_user(src, dirname=None):
    target_dir = dirname

    if target_dir is None:
        target_dir = inputDir + '/Unknown/' + time.strftime("%d_%H_%M_%S")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if len(os.listdir(target_dir)) > 256:
        target_dir = inputDir + '/Unknown/' + time.strftime("%d_%H_%M_%S")
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

    debug_print("Save unknown to " + target_dir)

    fileSeqNum = 1 + len(os.listdir(target_dir))
    faceFile = target_dir + "/face_" + str(fileSeqNum) + ".jpg"

    shutil.move(src, faceFile)

    return target_dir


def int_from_bytes(b3, b2, b1, b0):
    return (((((b3 << 8) + b2) << 8) + b1) << 8) + b0


def main():
    if redis_ready is False:
        debug_print('REDIS not ready.')
        return

    rds.set('frame', '1')

    cur_target_frame = -1
    next_target_frame = 1

    if not os.path.exists(inputDir + '/Unknown'):
        os.mkdir(inputDir + '/Unknown')

    dirname = inputDir + '/Unknown/' + time.strftime("%d_%H_%M_%S")

    if not os.path.exists(dirname):
        os.mkdir(dirname)

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

                    debug_print("recv frame: " + str(recv_frame) + ', x: ' + str(left) + ', y: ' + str(right) + ', t: ' + str(top) + ', b:' + str(bottom))

                    if cur_target_frame is -1:
                        cur_target_frame = recv_frame

                    debug_print("current frame: " + str(cur_target_frame))
                    debug_print("next frame: " + frame_str)

                    queued_frame = (int(frame_str) + 90 - recv_frame) % 90

                    info_print('delayed: ' + str(queued_frame))    

                    next_target_frame = int(frame_str)

                    if recv_frame == cur_target_frame:
                        fileName = "/tmp/input.jpg"
                        jpgFile = open(fileName, "wb")
                        jpgFile.write(ar)
                        jpgFile.close()

                        person, confidence = infer(fileName)

                        if confidence < threshold:
                            info_print("Who are you?: " + person + '(' + str(int(100*confidence)) + '%)')

                            if confidence < threshold:
                                save_unknown_user(fileName, dirname)

                            person = "Unknown"
                            confidence = 0.0

                        else:
                            info_print("{} : {:.2f} %".format(person, 100 * confidence))

                            if sock_ready is True:
                                b_array = bytes()
                                floatList = [left, right, top, bottom, confidence, label_list.index(person)]
                                debug_print("INDEX: " + str(label_list.index(person)))
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
                                info_print("Who are you?: " + person + '(' + str(int(100*confidence)) + '%)')

                                if confidence < threshold:
                                    dirname = save_unknown_user(fileName, dirname)

                                confidence = 0.0
                                person = "Unknown"

                            else:
                                info_print("{} : {:.2f} %".format(person, 100 * confidence))

                                if sock_ready is True:
                                    b_array = bytes()
                                    floatList = [left, right, top, bottom, confidence, label_list.index(person)]
                                    debug_print("INDEX: " + str(label_list.index(person)))
                                    b_array = b_array.join((struct.pack('f', val) for val in floatList))
                                    sock.send(b_array)

                        else:
                            cur_target_frame = next_target_frame
    except:
        debug_print('Exit')


if __name__ == "__main__":
    main()

