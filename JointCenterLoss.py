import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import openface
import redis
import socket
import array
import struct
import sys
import dlib
import argparse
import csv
import pickle


LAMBDA = 0.1
CENTER_LOSS_ALPHA = 0.5

representation_dim = 128
input_width = 96
input_height = 96
num_channel = 3
batch_size = 32
test_size = 100
num_class_per_group = 23  
num_epoch = 10

# Network Parameters
g_fc_layer1_dim = 1024
g_fc_layer2_dim = 512  # Final representation
g_fc_layer3_dim = 128

g_dense_block_layers = 4
g_dense_block_depth = 128

dlibDetector = dlib.get_frontal_face_detector()
align = openface.AlignDlib('openface/models/dlib/shape_predictor_68_face_landmarks.dat')
triplet = openface.TorchNeuralNet('openface/models/openface/nn4.small2.v1.t7', imgDim='96', cuda=True)


def get_center_loss(features, labels):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers')

    len_features = features.get_shape()[1]

    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.reduce_sum((features - centers_batch)**2, [1]))

    return loss


def update_centers(features, labels, alpha):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers')

    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers = tf.scatter_sub(centers, labels, diff)

    return centers


def fc_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='fc_network', bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        g_fc_layer1 = layers.fc(x, g_fc_layer1_dim, use_bias=False, scope='g_fc_layer1')
        g_fc_layer1 = layers.batch_norm(g_fc_layer1, bn_phaze, scope='g_fc_layer1_bn')
        g_fc_layer1 = act_func(g_fc_layer1)
        g_fc_layer1 = tf.nn.dropout(g_fc_layer1, keep_prob=keep_prob)

        g_fc_layer2 = layers.fc(g_fc_layer1, g_fc_layer2_dim, use_bias=False, scope='g_fc_layer2')
        g_fc_layer2 = layers.batch_norm(g_fc_layer2, bn_phaze, scope='g_fc_layer2_bn')
        g_fc_layer2 = act_func(g_fc_layer2)
        g_fc_layer2 = tf.nn.dropout(g_fc_layer2, keep_prob=keep_prob)

        g_fc_layer3 = layers.fc(g_fc_layer2, g_fc_layer3_dim, use_bias=False, scope='g_fc_layer3')
        g_fc_layer3 = layers.batch_norm(g_fc_layer3, bn_phaze, scope='g_fc_layer3_bn')
        g_fc_layer3 = act_func(g_fc_layer3)
        g_fc_layer3 = tf.nn.dropout(g_fc_layer3, keep_prob=keep_prob)

        return g_fc_layer3


def cnn_network(x, activation='swish', scope='cnn_network', reuse=False, bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        l = layers.conv(x, scope='conv1', filter_dims=[3, 3, 256], stride_dims=[1, 1], non_linear_fn=None, bias=False)

        with tf.variable_scope('dense_block_1'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_2'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_3'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                    scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_4'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, bn_phaze=bn_phaze,
                                           scope='layer' + str(i))
            l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                                  scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_5'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, bn_phaze=bn_phaze,
                                           scope='layer' + str(i))
            last_dense_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                                    scope='dense_transition_1', bn_phaze=bn_phaze)
        '''
        with tf.variable_scope('dense_block_6'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, use_bn=False,
                                    bn_phaze=bn_phaze, scope='layer' + str(i))
            l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                                    scope='dense_transition_1', use_bn=True, bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_7'):
            for i in range(g_dense_block_layers):
                l = layers.add_dense_layer(l, filter_dims=[3, 3, g_dense_block_depth], act_func=act_func, use_bn=False,
                                    bn_phaze=bn_phaze, scope='layer' + str(i))
            last_dense_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth], act_func=act_func,
                                                    scope='dense_transition_1', use_bn=True, bn_phaze=bn_phaze)
        '''

        return last_dense_layer


def get_triplet_representation(img):
    bgrImg = img
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    return triplet.forward(rgbImg)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        img = cv2.imread(fullname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # To RGB format
        if img is not None:
            images.append(np.array(img))
    return np.array(images)


def get_triplet_representation_align_image(img_file_path):
    bgrImg = cv2.imread(img_file_path)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #print('Dlib bbox detect')
    bbs = dlibDetector(rgbImg, 1)

    if len(bbs) == 0:
        #print('No bbox')
        return [], []

    '''
    reps = []

    for bb in bbs:
        # alignedFace = align.align(args.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP, skipMulti=True)
        alignedFace = align.align(96, rgbImg, bb, landmarkIndices=[8, 36, 45], skipMulti=True)

        rep = triplet.forward(alignedFace)
        reps.append((bb.center().x, rep))

    sreps = sorted(reps, key=lambda x: x[0])
    '''
    alignedFace = align.align(input_width, rgbImg, bbs[0], landmarkIndices=[8, 36, 45], skipMulti=True)

    rep = triplet.forward(alignedFace)

    alignedFace = np.array(alignedFace).reshape(-1, input_width, input_height, num_channel)

    return [rep], alignedFace # RGB format


def train(model_path):
    trX = []
    trY = []
    trXT = []

    teX = []
    teY = []
    teXT = []

    for idx, labelname in enumerate(os.listdir(imgs_dirname)):
        print('label:', idx, labelname)

        imgs_list = load_images_from_folder(os.path.join(imgs_dirname, labelname))

        for idx2, img in enumerate(imgs_list):
            label = np.zeros(len(os.listdir(imgs_dirname)))
            label[idx] += 1
            if idx2 < len(imgs_list) * 0.8:
                trX.append(img)
                trY.append(label)
                trXT.append(get_triplet_representation(img))
            else:
                teX.append(img)
                teY.append(label)
                teXT.append(get_triplet_representation(img))

    trX, trY, trXT = shuffle(trX, trY, trXT)

    trX = np.array(trX)
    trY = np.array(trY)
    trXT = np.array(trXT)
    teX = np.array(teX)
    teY = np.array(teY)
    teXT = np.array(teXT)

    trX = trX.reshape(-1, input_height, input_width, num_channel)
    teX = teX.reshape(-1, input_height, input_width, num_channel)

    X = tf.placeholder("float", [None, input_height, input_width, num_channel])
    Y = tf.placeholder("float", [None, num_class_per_group])
    TripletX = tf.placeholder("float", [None, representation_dim])
    bn_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # Network setup
    cnn_representation = cnn_network(X, bn_phaze=bn_train)
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, g_fc_layer3_dim)
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    fc_representation = cnn_representation

    #fc_representation = fc_network(cnn_representation, bn_phaze=bn_train)

    # Concat or Transform Here
    #representation = tf.concat([fc_representation, TripletX], 1)
    gamma = tf.get_variable("gamma", shape=[g_fc_layer3_dim], initializer=tf.constant_initializer(0.0))
    #gamma = tf.nn.softmax(gamma)  # Attention
    gamma = tf.nn.sigmoid(gamma)
    representation = tf.add(fc_representation, tf.multiply(gamma, TripletX))

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, g_fc_layer3_dim], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    center_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)
    entropy_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=0.1))

    total_loss = entropy_loss + center_loss * LAMBDA

    train_op = tf.train.AdamOptimizer(0.003).minimize(total_loss)

    predict_op = tf.argmax(tf.nn.softmax(prediction), 1)

    # Launch the graph in a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model loaded')
        except:
            print('Start New Training. Wait ...')

        num_itr = 0
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))

        for i in range(num_epoch):
            trX, trY, trXT = shuffle(trX, trY, trXT)

            for start, end in training_batch:
                _, c, g, center, _ = sess.run([train_op, entropy_loss, gamma, center_loss, update_center],
                                feed_dict={X: trX[start:end], Y: trY[start:end], TripletX: trXT[start:end], bn_train: True, keep_prob:0.5})
                num_itr = num_itr + 1

                if num_itr % 10 == 0:
                    try:
                        print('entropy loss: ' + str(c))
                        print('center loss: ' + str(center))
                        print('attention: ' + str(g))

                        saver.save(sess, model_path)
                    except:
                        print('Save failed')

            test_indices = np.arange(len(teX))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            print('# Test Set #')
            print(np.argmax(teY[test_indices], axis=1))

            print('# Prediction #')
            print(sess.run(predict_op,
                           feed_dict={X: teX[test_indices], Y: teY[test_indices], TripletX: teXT[test_indices], bn_train: False, keep_prob:1.0}))

            precision = np.mean(np.argmax(teY[test_indices], axis=1) ==
                             sess.run(predict_op,
                                      feed_dict={X: teX[test_indices], Y: teY[test_indices], TripletX: teXT[test_indices], bn_train: False, keep_prob:1.0}))
            print('epoch ' + str(i) + ', precision: ' + str(100 * precision) + ' %')


def int_from_bytes(b3, b2, b1, b0):
    return (((((b3 << 8) + b2) << 8) + b1) << 8) + b0


def test(model_path):
    print('Test Mode')

    X = tf.placeholder("float", [None, input_height, input_width, num_channel])
    Y = tf.placeholder("float", [None, num_class_per_group])
    TripletX = tf.placeholder("float", [None, representation_dim])
    bn_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # Network setup
    cnn_representation = cnn_network(X, bn_phaze=bn_train)
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim)
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    fc_representation = cnn_representation

    # fc_representation = fc_network(cnn_representation, bn_phaze=bn_train)

    # Concat or Transform Here
    # representation = tf.concat([fc_representation, TripletX], 1)
    gamma = tf.get_variable("gamma", shape=[representation_dim], initializer=tf.constant_initializer(0.0))
    #gamma = tf.nn.softmax(gamma)  # Attention
    gamma = tf.nn.sigmoid(gamma)
    representation = tf.add(fc_representation, tf.multiply(gamma, TripletX))

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, g_fc_layer3_dim], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    center_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)
    entropy_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=0.1))

    total_loss = entropy_loss + center_loss * LAMBDA

    train_op = tf.train.AdamOptimizer(0.003).minimize(total_loss)

    predict_op = tf.argmax(tf.nn.softmax(prediction), 1)
    confidence_op = tf.nn.softmax(prediction)

    # Launch the graph in a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model loaded')
        except:
            print('Model load failed: ' + model_path)
            return

        fileDir = os.path.dirname(os.path.realpath(__file__))
        # Modify baseDir to your environment
        inputDir = fileDir + '/input'
        label_list = [d for d in os.listdir(inputDir + '/user') if os.path.isdir(inputDir + '/user/' + d)]
        label_list.sort(key=str.lower)

        print(label_list)  # Global lable list.

        group_label_list = os.listdir(imgs_dirname)

        redis_ready = False

        clf_directory = os.path.dirname(os.path.realpath(__file__)) + '/svm/group/'
        clf_files = os.listdir(clf_directory)
        clf_list = [pickle.load(open(clf_directory + pkl_file, 'r')) for pkl_file in clf_files]

        try:
            rds = redis.StrictRedis(host=REDIS_SERVER, port=REDIS_PORT, db=0)

            p = rds.pubsub()
            p.subscribe(redis_channel)
            redis_ready = True

            print('Connected to Message Queue')
        except:
            redis_ready = False
            print('Faile to connect to Message Queue')

        sock_ready = False

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            sock_ready = True

            print('Connected to Edge Camera')
        except:
            print('Faile to connect to Edge Camera')
            sock_ready = False

        if redis_ready is False:
            print
            'REDIS not ready.'
            return

        cur_target_frame = -1
        next_target_frame = 1

        if not os.path.exists(inputDir + '/../Unknown'):
            os.mkdir(inputDir + '/../Unknown')

        dirname = inputDir + '/../Unknown'

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        for item in p.listen():
            data = item

            if data is not None:
                data = data.get('data')

                if data != 1:
                    temp = array.array('B', data)
                    ar = np.array(temp, dtype=np.uint8)

                    left = int_from_bytes(ar[4], ar[3], ar[2], ar[1])
                    right = int_from_bytes(ar[8], ar[7], ar[6], ar[5])
                    top = int_from_bytes(ar[12], ar[11], ar[10], ar[9])
                    bottom = int_from_bytes(ar[16], ar[15], ar[14], ar[13])

                    recv_frame = ar[0]

                    ar = ar[17:]

                    frame_str = rds.get(frame_db)

                    if cur_target_frame is -1:
                        cur_target_frame = recv_frame

                    next_target_frame = int(frame_str)

                    if recv_frame == cur_target_frame or recv_frame == next_target_frame:
                        fileName = '/tmp/input' + redis_channel + '.jpg'
                        jpgFile = open(fileName, "wb")
                        jpgFile.write(ar)
                        jpgFile.close()

                        confidence = 0.97
                        person = 'Unknown'

                        #print('Get triplet representation')
                        tpReps, img = get_triplet_representation_align_image(fileName)

                        if not tpReps:
                            print('Not a valid face.')
                        else:
                            #print('Run prediction..')
                            use_softmax = False
                            pred_id, confidence, rep = sess.run([predict_op, confidence_op, representation],
                                                                feed_dict={X: img, TripletX: tpReps, bn_train: False,
                                                                           keep_prob: 1.0})

                            if use_softmax is True:
                                #print('# Prediction: ' + str(pred_id))
                                person = group_label_list[pred_id[0]]
                                confidence = confidence[0][pred_id[0]]
                                print('# Person: ' + person + ', Confidence: ' + str(confidence))
                            else:
                                confidences = []
                                labels = []

                                for (le, clf) in clf_list:
                                    pred = clf.predict_proba(rep).ravel()
                                    maxI = np.argmax(pred)
                                    person = le.inverse_transform([maxI])
                                    confidence = pred[maxI]
                                    confidences.append(confidence)
                                    labels.append(person[0])

                                print('#################################')
                                print(labels)
                                print(confidences)

                                effective_labels = []
                                effective_confidences = []

                                for i in range(len(labels)):
                                    if labels[i] != 'Unknown':
                                        effective_labels.append(labels[i])
                                        effective_confidences.append(confidences[i])

                                if len(effective_labels) == 0:
                                    person = 'Unknown'
                                    confidence = 0.99
                                else:
                                    confidence = max(effective_confidences)
                                    maxI = effective_confidences.index(confidence)
                                    person = effective_labels[maxI]

                                    '''
                                    if len(effective_labels) > 1:
                                        effective_confidences.sort(reverse=True)

                                        if effective_confidences[0] - effective_confidences[1] < 0.25:
                                            person = 'Unknown'
                                            confidence = 0.99
                                    '''
                                print('\nPerson: ' + person + ', Confidence: ' + str(confidence * 100) + '%')

                            if sock_ready is True:
                                if person != 'Unknown' and person != 'Nobody':
                                    b_array = bytes()
                                    floatList = [left, right, top, bottom, confidence, label_list.index(person)]
                                    b_array = b_array.join((struct.pack('f', val) for val in floatList))
                                    sock.send(b_array)
                    else:
                        cur_target_frame = next_target_frame

                else:
                    rds.set(frame_db, '1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test/reps', default='train')
    parser.add_argument('--redis_host', type=str, help='redis server address', default='10.144.164.204')
    parser.add_argument('--redis_port', type=int, default=8090, help='redis server port')
    parser.add_argument('--camera_host', type=str, help='edge camera server address', default='10.144.164.174')
    parser.add_argument('--camera_port', type=int, default=6000, help='edge camera server port')
    parser.add_argument('--redis_camera_channel', type=str, help='camera input channel', default='camera1')
    parser.add_argument('--redis_frame_channel', type=str, help='frame control channel', default='frame1')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--data', type=str, help='data source base directory', default='./input')
    parser.add_argument('--out', type=str, help='output directory', default='./out/embedding')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--label', type=str, help='training data directory', default='input')

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    imgs_dirname = args.train_data
    label_directory = args.label

    if mode == 'train':
        num_class_per_group = len(os.listdir(imgs_dirname))
        train(model_path)
    elif mode == 'test':
        num_class_per_group = len(os.listdir(label_directory))
        HOST = args.camera_host
        PORT = args.camera_port
        REDIS_SERVER = args.redis_host
        REDIS_PORT = args.redis_port
        redis_channel = args.redis_camera_channel
        frame_db = args.redis_frame_channel

        test(model_path)
    elif mode == 'embedding':
        out_dir = args.out

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data_dir = args.data

        num_class_per_group = len(os.listdir(label_directory))

        if not os.path.exists(data_dir):
            print('No data.')
        else:
            X = tf.placeholder("float", [None, input_height, input_width, num_channel])
            TripletX = tf.placeholder("float", [None, representation_dim])
            bn_train = tf.placeholder(tf.bool)
            keep_prob = tf.placeholder(tf.float32)

            # Network setup
            cnn_representation = cnn_network(X, bn_phaze=bn_train)
            print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

            cnn_representation = layers.global_avg_pool(cnn_representation, g_fc_layer3_dim)
            print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

            fc_representation = cnn_representation

            # fc_representation = fc_network(cnn_representation, bn_phaze=bn_train)

            # Concat or Transform Here
            # representation = tf.concat([fc_representation, TripletX], 1)
            gamma = tf.get_variable("gamma", shape=[g_fc_layer3_dim], initializer=tf.constant_initializer(0.0))
            #gamma = tf.nn.softmax(gamma)
            gamma = tf.nn.sigmoid(gamma)
            representation = tf.add(fc_representation, tf.multiply(gamma, TripletX))

            prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

            with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
                centers = tf.get_variable('centers', [num_class_per_group, g_fc_layer3_dim], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0), trainable=False)

            # Launch the graph in a session
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                try:
                    saver = tf.train.Saver()
                    saver.restore(sess, model_path)
                    print('Model loaded')
                except:
                    print('Model load failed: ' + model_path)

                label_f = open(os.path.join(out_dir, 'labels.csv'), mode='w')
                label_csv = csv.writer(label_f)

                embedding_f = open(os.path.join(out_dir, 'reps.csv'), mode='w')
                embedding_csv = csv.writer(embedding_f)

                for idx, labelname in enumerate(os.listdir(data_dir)):
                    image_directory = os.path.join(data_dir, labelname)

                    for filename in os.listdir(image_directory):
                        fullname = os.path.join(image_directory, filename).replace("\\", "/")
                        label_csv.writerow([idx+1, fullname])

                        img = cv2.imread(fullname)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # To RGB format
                        tpReps = get_triplet_representation(img)

                        reps = sess.run(representation, feed_dict={X: [img], TripletX: [tpReps], bn_train: False, keep_prob: 1.0})

                        reps = np.squeeze(reps).tolist()
                        embedding_csv.writerow(reps)

                label_f.close()
                embedding_f.close()





