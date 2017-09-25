#!/usr/bin/env python
import redis
import array
import argparse
import openface
import cv2
import pickle
import time
import ast
import shutil
import socket
import struct
import dlib
from sklearn.mixture import GMM
from scipy.spatial import distance
import os
import pandas as pd
from operator import itemgetter
import numpy as np
import signal
import sys
import random
from nolearn.dbn import DBN
from sklearn.metrics import log_loss


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("recognizer.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def debug_print(str):
    if debug is True:
        print(str)


def info_print(str):
    if info is True:
        print(str)

def dbn_loss_func(targets, outputs):
    pass

def getRep(imgPath, multiple=False):

    if show_time is True:
        start = time.time()

    bgrImg = cv2.imread(imgPath)

    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #for test
    if use_face_align is False:
        rgbImg = cv2.resize(rgbImg, (96, 96))
        rep = net.forward(rgbImg)
        rep = np.asarray(rep)
        reps = []
        reps.append(rep)
        return reps

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
        #debug_print(bb1)


    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
        ##debug_print("Unable to find a face")
        return []
    """

    if show_time is True:
        debug_print("BBox took {} seconds.".format(time.time() - start))

    reps = []

    for bb in bbs:
        if show_time is True:
            start = time.time()

       #alignedFace = align.align(args.imgDim, rgbImg, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        alignedFace = align.align(args.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

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


def infer(fileName, mode):
    represent = getRep(fileName, False)
    confidence = 0.0

    if not represent:
        ##debug_print("Who are you?")
        return 'Unknown', confidence

    #for test
    if use_face_align is False:
        r = represent[0]
        rep = r.reshape(1, -1)
    else:
        r = represent[0]
        rep = r[1].reshape(1, -1)
        bbx = r[0]

    if save_representation is True:
        save_rep(fileName, rep)

    if show_time is True:
        start = time.time()

    confidence = 0
    confidence_dbn = 0
    person = ''
    person_dbn = ''
    std_list = {}
    mean_list = {}
    emb_list = {}

    if mode == 'user':
        #info_print('User Group')
        predictions = clf.predict_proba(rep).ravel()
        pred_dbn = clf_dbn.predict_proba(rep).ravel()

        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        maxI = np.argmax(pred_dbn)
        person_dbn = le_dbn.inverse_transform(maxI)
        confidence_dbn = pred_dbn[maxI]
        mean_list = g_dist_mean_list
        std_list = g_std_list
        emb_list = g_embedding_list

    else:
        #info_print('IGuest Group')
        predictions = clf_iguest.predict_proba(rep).ravel()
        pred_dbn = clf_dbn_iguest.predict_proba(rep).ravel()

        maxI = np.argmax(predictions)
        person = le_iguest.inverse_transform(maxI)
        confidence = predictions[maxI]

        maxI = np.argmax(pred_dbn)
        person_dbn = le_dbn_iguest.inverse_transform(maxI)
        confidence_dbn = pred_dbn[maxI]
        mean_list = g_iguest_dist_mean_list
        std_list = g_iguest_std_list
        emb_list = g_iguest_embedding_list
    '''
    else:
        #info_print('OGuest Group')
        predictions = clf_oguest.predict_proba(rep).ravel()
        pred_dbn = clf_dbn_oguest.predict_proba(rep).ravel()

        maxI = np.argmax(predictions)
        person = le_oguest.inverse_transform(maxI)
        confidence = predictions[maxI]

        maxI = np.argmax(pred_dbn)
        person_dbn = le_dbn_oguest.inverse_transform(maxI)
        confidence_dbn = pred_dbn[maxI]
        mean_list = g_oguest_dist_mean_list
        std_list = g_oguest_std_list
        emb_list = g_oguest_embedding_list
    '''

    print "\n   DBN: ", person_dbn, confidence_dbn
    print "   SVM: ", person, confidence

    if person != person_dbn:
        return 'Unknown', 0

    #if confidence < 0.85:  # Hard limit
    #    return person, confidence
    
    leaky = 0.7

    #c = np.array([confidence_dbn, confidence])

    #avgt = np.mean([confidence_dbn, confidence])

    avgt = leaky * confidence_dbn + (1 - leaky) * confidence
    confidence = avgt

    print '   AVG: ', avgt

    margin = 0 #0.05

    if mode == 'user':
        if avgt < threshold + margin:
            #if confidence > fn_threshold:
            if avgt > fn_threshold:
                dist_list = []

                #print person, confidence

                sz = len(emb_list[person])

                for i in range(sz):
                    dst = distance.euclidean(emb_list[person][i], rep)
                    dist_list.append(dst)

                m = np.mean(dist_list)
                #m = np.max(dist_list)

                confidence_interval = std_list[person]

                thd = mean_list[person] + confidence_interval

                print '   DIST: ', m, thd

                c = np.array([confidence_dbn, confidence])

                if m < thd:
                    confidence = avgt   #c.max()
                else:
                    confidence = c.min()

        else:
            confidence = avgt
    else:
        c = np.array([confidence_dbn, confidence])

        if person == person_dbn:
            avgt = c.max()
        else:
            avgt = confidence

        if avgt < threshold + margin:
            if avgt > 0.5:  #loose limit for guest group
                dist_list = []

                # print person, confidence

                sz = len(emb_list[person])

                for i in range(sz):
                    dst = distance.euclidean(emb_list[person][i], rep)
                    dist_list.append(dst)

                m = np.mean(dist_list)
                # m = np.max(dist_list)

                confidence_interval = std_list[person]

                thd = mean_list[person] + confidence_interval

                print '   DIST: ', m, thd

                if m < thd:
                    confidence = threshold # Boost to threshold
                else:
                    confidence = avgt

        else:
            confidence = avgt

    return person, confidence


def save_rep(imgFile, rep, dirname=None):
    if not os.path.exists(repDir):
        os.mkdir(repDir)

    target_dir = dirname

    if target_dir is None:
        target_dir = repDir + '/' + time.strftime("%d_%H_%M_%S")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if len(os.listdir(target_dir)) > 256:
        target_dir = repDir + '/' + time.strftime("%d_%H_%M_%S")
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

    #debug_print("Save unknown to " + target_dir)

    fileSeqNum = 1 + len(os.listdir(target_dir))
    faceFile = target_dir + "/face_" + str(fileSeqNum) + ".jpg"

    shutil.copy2(imgFile, faceFile)

    repFileName = target_dir + "/face_" + str(fileSeqNum) + "_rep.txt"
    repFile = open(repFileName, "w")
    repFile.write(str(rep))
    repFile.close()


def save_unknown_user(src, dirname=None, candidate=None):
    target_dir = dirname
    name = candidate

    if candidate is None:
        name = 'face'

    '''
    if target_dir is None:
        target_dir = inputDir + '/../Unknown/' + time.strftime("%d_%H_%M_%S")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if len(os.listdir(target_dir)) > 256:
        target_dir = inputDir + '/../Unknown/' + time.strftime("%d_%H_%M_%S")
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

    #print("Save unknown to " + target_dir)
    '''

    if not os.path.exists(target_dir + '/' + name):
        os.mkdir(target_dir + '/' + name)

    fileSeqNum = 1 + len(os.listdir(target_dir + '/' + name))
    faceFile = target_dir + "/" + name + "/" + name + "_" + time.strftime("%d_%H_%M_%S_") + str(fileSeqNum) + ".jpg"

    shutil.move(src, faceFile)

    return target_dir


def int_from_bytes(b3, b2, b1, b0):
    return (((((b3 << 8) + b2) << 8) + b1) << 8) + b0


def initialize_rep_distance_list(embedding_path, min_list, mean_list, std_list, emb_list):
    embeddings_dir = embedding_path
    fname = "{}/labels.csv".format(embeddings_dir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]

    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.

    # print labels

    fname = "{}/reps.csv".format(embeddings_dir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    first_label = labels[0]

    min_means = []
    mean_means = []
    class_std = []
    class_labels = []
    classes = []

    embedding_list = []
    ems = []

    for i in range(len(labels)):
        if labels[i] == first_label:
            classes.append(labels[i])
            ems.append(embeddings[i])

        else:
            class_labels.append(classes)
            embedding_list.append(ems)
            classes = [labels[i]]
            ems = [embeddings[i]]
            first_label = labels[i]

    class_labels.append(classes)
    embedding_list.append(ems)

    for i in range(len(embedding_list)):
        if len(embedding_list[i]) > 100:
            random.shuffle(embedding_list[i])
            embedding_list[i] = embedding_list[i][:100]
            class_labels[i] = class_labels[i][:100]
            #print len(embedding_list[i]), len(class_labels[i])

    # print len(class_labels)

    for c in range(len(class_labels)):
        class_mean = []

        for p in range(len(class_labels[c])):
            dist_list = []

            for i in range(len(class_labels[c])):
                if i == p:
                    continue

                dst = distance.euclidean(embedding_list[c][i], embedding_list[c][p])
                dist_list.append(dst)

            m = np.mean(dist_list)

            class_mean.append(m)

        mm = np.mean(class_mean)
        m = np.min(class_mean)
        s = np.std(class_mean)
        min_means.append(m)
        mean_means.append(mm)
        class_std.append(s)

    for c in range(len(class_labels)):
        min_list[class_labels[c][0]] = min_means[c]
        mean_list[class_labels[c][0]] = mean_means[c]
        # g_std_list[class_labels[c][0]] = class_std[c]
        std_list[class_labels[c][0]] = alpha * class_std[c] / np.sqrt(len(class_labels[c]))

        print class_labels[c][0], len(class_labels[c]), std_list[class_labels[c][0]], mean_list[class_labels[c][0]], min_list[class_labels[c][0]]

    for c in range(len(class_labels)):
        emb_list[class_labels[c][0]] = embedding_list[c]


def handler(signum, frame):
    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, os_handler)

    if os.path.exists(inputDir + '/../on_training'):
        print '\n\nOn Training. Wait.'
    else:
        print '\n\nGood Bye.'
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)


def main():
    if redis_ready is False:
        #debug_print('REDIS not ready.')
        return

    cur_target_frame = -1
    next_target_frame = 1

    if not os.path.exists(inputDir + '/../Unknown'):
        os.mkdir(inputDir + '/../Unknown')

    #dirname = inputDir + '/../Unknown/' + time.strftime("%d_%H_%M_%S")
    dirname = inputDir + '/../Unknown'

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    embeddings_dir = os.path.dirname(os.path.realpath(__file__)) + '/../face_register/output/embedding/user'

    initialize_rep_distance_list(embeddings_dir, g_dist_min_list, g_dist_mean_list, g_std_list, g_embedding_list)

    embeddings_dir = os.path.dirname(os.path.realpath(__file__)) + '/../face_register/output/embedding/iguest'

    initialize_rep_distance_list(embeddings_dir, g_iguest_dist_min_list, g_iguest_dist_mean_list, g_iguest_std_list, g_iguest_embedding_list)

    #embeddings_dir = os.path.dirname(os.path.realpath(__file__)) + '/../face_register/output/embedding/oguest'

    #initialize_rep_distance_list(embeddings_dir, g_oguest_dist_min_list, g_oguest_dist_mean_list, g_oguest_std_list, g_oguest_embedding_list)

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

                    bbox_size = (right - left) * (bottom - top);

                    recv_frame = ar[0]

                    ar = ar[17:]

                    frame_str = rds.get('frame')

                    #debug_print("recv frame: " + str(recv_frame) + ', x: ' + str(left) + ', y: ' + str(right) + ', t: ' + str(top) + ', b:' + str(bottom))

                    if cur_target_frame is -1:
                        cur_target_frame = recv_frame

                    #debug_print("current frame: " + str(cur_target_frame))
                    #debug_print("next frame: " + frame_str)

                    queued_frame = (int(frame_str) + 30 - recv_frame) % 30

                    #info_print('delayed: ' + str(queued_frame))

                    next_target_frame = int(frame_str)

                    if recv_frame == cur_target_frame or recv_frame == next_target_frame:
                        fileName = "/tmp/input.jpg"
                        jpgFile = open(fileName, "wb")
                        jpgFile.write(ar)
                        jpgFile.close()

                        person, confidence = infer(fileName, 'user')

                        info_print("\n   User Group:" + person + '(' + str(int(100 * confidence)) + '%)')

                        candidate = person
                        candidate_confidence = confidence

                        '''
                        if candidate_confidence < threshold:
                            person, confidence = infer(fileName, 'iguest')
                            info_print("\n   IGuest Group:" + person + '(' + str(int(100 * confidence)) + '%)')

                            if candidate == 'Guest':
                                if person == 'User':
                                    candidate = 'Unknown'
                                else:
                                    if confidence >= threshold:
                                        candidate = person
                                        candidate_confidence = confidence
                                    elif confidence > fn_threshold and confidence >= candidate_confidence:
                                        candidate = person
                                        candidate_confidence = confidence

                            elif candidate != 'Unknown' and candidate != 'Nobody':
                                if person != 'User':
                                    if confidence >= threshold:
                                        candidate = person
                                        candidate_confidence = confidence
                                        # else:
                                        #    if confidence > candidate_confidence:
                                        #        candidate_confidence = confidence
                       
                        else:
                            person, confidence = infer(fileName, 'iguest')

                            info_print("\n   IGuest Group:" + person + '(' + str(int(100 * confidence)) + '%)')

                            if candidate == 'Guest':
                                if person == 'User':
                                    candidate = 'Unknown'
                                else:
                                    if confidence >= threshold:
                                        candidate = person
                                        candidate_confidence = confidence
                        '''

                        '''
                        if confidence < threshold:
                            person, confidence = infer(fileName, 'oguest')
                            info_print("OGuest Group:" + person + '(' + str(int(100 * confidence)) + '%)')

                            if confidence > candidate_confidence:
                                candidate = person
                                candidate_confidence = confidence
                        '''

                        if candidate_confidence < threshold:
                            info_print(
                                "\nWho are you?: " + candidate + '(' + str(int(100 * candidate_confidence)) + '%)')

                            save_unknown_user(fileName, dirname, candidate)

                            '''
                            if candidate_confidence > 0.8:
                                dirname = save_unknown_user(fileName, dirname, candidate)
                            else:
                                dirname = save_unknown_user(fileName, dirname, 'Unknown')
                            '''
                        else:
                            # info_print("{} : {} %, size : {}".format(person, int(100 * confidence), str(bbox_size)))
                            info_print("\nPredict {} : {} %".format(candidate, int(100 * candidate_confidence)))

                        if candidate == 'Guest':
                            candidate = 'Unknown'

                        if candidate_confidence > 0:
                            if sock_ready is True:
                                if candidate != 'Unknown' and candidate != 'Nobody':
                                    b_array = bytes()
                                    floatList = [left, right, top, bottom, candidate_confidence,
                                                 label_list.index(candidate)]
                                    # debug_print("INDEX: " + str(label_list.index(person)))
                                    b_array = b_array.join((struct.pack('f', val) for val in floatList))
                                    sock.send(b_array)
                    else:
                        cur_target_frame = next_target_frame

                else:
                    rds.set('frame', '1')
    except Exception, e:
        print 'Exit: ', str(e)


if __name__ == "__main__":
    os_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)

    sys.stdout = Logger()

    threshold = 0.95
    fn_threshold = 0.8
    alpha = 1.64  # 95%: 1.96, 90 %: 1.64
    show_time = False
    debug = False
    info = True
    save_representation = False
    use_face_align = True

    fileDir = os.path.dirname(os.path.realpath(__file__))
    # Modify baseDir to your environment
    baseDir = fileDir + '/../'
    modelDir = os.path.join(fileDir, baseDir + '/openface', 'models')
    inputDir = baseDir + 'face_register/input'
    repDir = baseDir + 'face_register/reps'
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')
    haarCascadeModelDir = '/usr/local/share/OpenCV/haarcascades/'
    dlibDetector = dlib.get_frontal_face_detector()
    label_list = [d for d in os.listdir(inputDir + '/user') if os.path.isdir(inputDir + '/user/' + d)]
    label_list.sort()

    label_list_iguest = [d for d in os.listdir(inputDir + '/iguest') if os.path.isdir(inputDir + '/iguest/' + d)]
    label_list_iguest.sort()

    # label_list_oguest = [d for d in os.listdir(inputDir + '/oguest')]
    # label_list_oguest.sort()

    label_list.extend(label_list_iguest)
    # label_list.extend(label_list_oguest)

    print(label_list)

    g_dist_min_list = {}
    g_dist_mean_list = {}
    g_std_list = {}
    g_embedding_list = {}

    g_iguest_dist_min_list = {}
    g_iguest_dist_mean_list = {}
    g_iguest_std_list = {}
    g_iguest_embedding_list = {}

    g_oguest_dist_min_list = {}
    g_oguest_dist_mean_list = {}
    g_oguest_std_list = {}
    g_oguest_embedding_list = {}

    HOST, PORT = "127.0.0.1", 55555
    # HOST, PORT = "10.100.0.53", 55555
    REDIS_SERVER = '127.0.0.1'
    REDIS_PORT = 6379
    parser = argparse.ArgumentParser()

    # parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
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

    try:
        rds = redis.StrictRedis(host=REDIS_SERVER, port=REDIS_PORT, db=0)
        
        p = rds.pubsub()
        p.subscribe('camera')
        redis_ready = True

    except:
        redis_ready = False

    (le, clf) = pickle.load(open(baseDir + '/svm/classifier.pkl', 'r'))
    (le_dbn, clf_dbn) = pickle.load(open(baseDir + '/dbn/classifier.pkl', 'r'))

    # Individual Guest Group
    (le_iguest, clf_iguest) = pickle.load(open(baseDir + '/svm/classifier_iguest.pkl', 'r'))
    (le_dbn_iguest, clf_dbn_iguest) = pickle.load(open(baseDir + '/dbn/classifier_iguest.pkl', 'r'))

    # Overall Guest Group
    # (le_oguest, clf_oguest) = pickle.load(open(baseDir + '/svm/classifier_oguest.pkl', 'r'))
    # (le_dbn_oguest, clf_dbn_oguest) = pickle.load(open(baseDir + '/dbn/classifier_oguest.pkl', 'r'))

    sock_ready = False

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sock_ready = True

        # debug_print('Connected')
    except:
        sock_ready = False

    main()

