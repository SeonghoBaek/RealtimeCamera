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
import shutil


LAMBDA = 1e-3
GAMMA = 1.0

CENTER_LOSS_ALPHA = 0.5
DISTANCE_MARGIN = 10.0

representation_dim = 128
input_width = 96
input_height = 96
scale_size = 112
num_channel = 3
num_patch = 4
batch_size = 16
test_size = 100
num_class_per_group = 46
num_epoch = 50

# Network Parameters
g_fc_layer1_dim = 1024
g_fc_layer2_dim = 512  # Final representation
g_fc_layer3_dim = 128

g_dense_block_layers = 4
g_dense_block_depth = 128

lstm_hidden_size_layer1 = 128
lstm_hidden_size_layer2 = 128
lstm_sequence_length = 96
lstm_representation_dim = 128

dlibDetector = dlib.get_frontal_face_detector()
align = openface.AlignDlib('openface/models/dlib/shape_predictor_68_face_landmarks.dat')
#triplet = openface.TorchNeuralNet('openface/models/openface/nn4.small2.v1.t7', imgDim='96', cuda=True)

X = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
Y = tf.placeholder(tf.float32, [None, num_class_per_group])
LSTM_X = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, lstm_sequence_length])
TripletX = tf.placeholder(tf.float32, [None, representation_dim])
bn_train = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def get_center_loss(features, labels):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers')

    len_features = features.get_shape()[1]

    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.reduce_sum((features - centers_batch)**2, [1]))

    # Center distance loss
    shuffle_labels = tf.random.shuffle(labels)
    shuffle_centers = tf.gather(centers, shuffle_labels)

    distance_loss = DISTANCE_MARGIN / tf.reduce_mean(tf.reduce_sum((centers_batch - shuffle_centers)**2, [1]))

    return loss, distance_loss


def update_centers(features, labels, alpha):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers')

    labels = tf.reshape(labels, [-1]) # flatten
    centers_batch = tf.gather(centers, labels) # Gather center tensor by labels value order
    diff = centers_batch - features # L1 distance array between each of center and feature

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers = tf.scatter_sub(centers, labels, diff)

    return centers


def fc_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='fc_network', bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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


def lstm_network(input_data, scope='lstm_network', forget_bias=1.0, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer1/2)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer1/2)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

        _, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)

        #print(states)
        #print(states[0])
        #print(states[1])

        states_concat = tf.concat([states[0].h, states[1].h], 1)

        print('LSTM Representation Dimension: ' + str(states_concat.get_shape().as_list()))

    return states_concat


def decoder_network(x, anchor_layer=None, activation='swish', scope='g_decoder_network', bn_phaze=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        print('decoder input:', str(x.get_shape().as_list()))
        l = tf.reshape(x, shape=[-1, 4, 4, 18]) # 256 feature

        # 6 x 6
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv1', filter_dims=[3, 3, 512],
                             stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        print('deconv1:', str(l.get_shape().as_list()))

        # 12 x 12
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv2', filter_dims=[3, 3, g_dense_block_depth],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)
        print('deconv2:', str(l.get_shape().as_list()))

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = act_func(l)

        # 24 x 24
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv3', filter_dims=[3, 3, g_dense_block_depth],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)
        print('deconv3:', str(l.get_shape().as_list()))

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        l = act_func(l)
        # 48 x 48
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv4', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        print('deconv4:', str(l.get_shape().as_list()))

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        if anchor_layer is not None:
            #anchor_layer = anchor_layer + tf.random_normal(shape=tf.shape(anchor_layer), mean=0.0, stddev=1.0, dtype=tf.float32)
            l = tf.concat([l, anchor_layer], axis=3)

        l = act_func(l)
        # 96 x 96
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv5', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, 3], act_func=act_func,
                                              scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        l = act_func(l)

        print('deconv5:', str(l.get_shape().as_list()))

        return l


def discriminator(input_data, activation='swish', scope='discriminator', reuse=False, bn_phaze=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #if reuse:
        #    tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'tanh':
            act_func = tf.nn.tanh
        else:
            act_func = tf.nn.sigmoid

        l = layers.conv(input_data, scope='conv1', filter_dims=[3, 3, g_dense_block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_0')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_6')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7')

        # dc_final_layer = batch_norm_conv(last_dense_layer, b_train=bn_phaze, scope='last_dense_layer')

        l = layers.global_avg_pool(l, representation_dim)
        dc_final_layer = l

        dc_output = layers.fc(dc_final_layer, scope='g_enc_z_fc', out_dim=1, non_linear_fn=None)

    return dc_final_layer, dc_output, tf.sigmoid(dc_output)


def add_residual_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, bn_phaze=False, scope='residual_dense_block'):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = input_dims[-1]

        for i in range(num_layers):
            l = layers.add_residual_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i))

        l = tf.add(l, in_layer)

    return l


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, bn_phaze=False, scope='residual_dense_block'):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        for i in range(num_layers):
            l = layers.add_dense_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i))
        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_out], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        l = tf.add(l, in_layer)

    return l


def encoder_network(x, activation='relu', scope='encoder_network', reuse=False, bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #if reuse:
        #    tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        l = layers.conv(x, scope='conv1', filter_dims=[3, 3, g_dense_block_depth], stride_dims=[1, 1], non_linear_fn=None, bias=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_0')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        l_share = l

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_6')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_8')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_9')

        with tf.variable_scope('dense_block_last'):
            scale_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth],
                                                            act_func=act_func,
                                                            scope='dense_transition_1', bn_phaze=bn_phaze,
                                                            use_pool=False)
            last_dense_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth],
                                                                 act_func=act_func,
                                                                 scope='dense_transition_2', bn_phaze=bn_phaze,
                                                                 use_pool=False)
            scale_layer = act_func(scale_layer)
            last_dense_layer = act_func(last_dense_layer)

    return last_dense_layer, scale_layer, l_share


'''
def get_triplet_representation(img, to_rgb=False):
    if to_rgb is True:
        bgrImg = img
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    else:
        rgbImg = img

    return triplet.forward(rgbImg)
'''

def load_images_patch(filename, b_align=False):
    images = []
    lstm_images = []

    if b_align == True:
        img = get_align_image(filename)

        if len([img]) == 0:
            return []
    else:
        jpg_img = cv2.imread(filename)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)
        grey_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)

    if img is not None:
        img = np.array(img)
        images.append(img)

        lstm_image = cv2.resize(grey_img, dsize=(lstm_sequence_length, lstm_sequence_length))
        lstm_images.append(lstm_image)

        img = cv2.resize(img, dsize=(scale_size, scale_size))
        grey_img = cv2.resize(grey_img, dsize=(scale_size, scale_size))

        #croped = img[scale_size / 2 - input_width / 2:scale_size / 2 + input_width / 2,
        #         scale_size / 2 - input_width / 2:scale_size / 2 + input_width / 2].copy()

        #images.append(croped)

        dy = np.random.random_integers(low=1, high=img.shape[0]-input_height, size=num_patch-1)
        dx = np.random.random_integers(low=1, high=img.shape[1]-input_width, size=num_patch-1)

        window = zip(dy, dx)

        for i in range(len(window)):
            croped = img[window[i][0]:window[i][0]+input_height, window[i][1]:window[i][1]+input_width].copy()
            #cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)
            images.append(croped)

            croped_grey = grey_img[window[i][0]:window[i][0] + input_height,
                          window[i][1]:window[i][1] + input_width].copy()
            #lstm_image = cv2.resize(croped_grey, dsize=(lstm_sequence_length, lstm_sequence_length))
            lstm_images.append(lstm_image)

    images = np.array(images)
    lstm_images = np.array(lstm_images)

    return images, lstm_images


def load_images_from_folder(folder):
    images = []
    lstm_images = []

    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB) # To RGB format
        grey_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)

        if img is not None:
            img = np.array(img)
            grey_img = np.array(grey_img)

            images.append(img)

            lstm_image = cv2.resize(grey_img, dsize=(lstm_sequence_length, lstm_sequence_length))
            lstm_images.append(lstm_image)

            #scale = scale_size / input_width

            img = cv2.resize(img, dsize=(scale_size, scale_size))
            grey_img = cv2.resize(grey_img, dsize=(scale_size, scale_size))

            #croped = img[scale_size/2 - input_width/2:scale_size/2 + input_width/2, scale_size/2 - input_width/2:scale_size/2 + input_width/2].copy()

            #images.append(croped)

            dy = np.random.random_integers(low=1, high=img.shape[0]-input_height, size=num_patch-1)
            dx = np.random.random_integers(low=1, high=img.shape[1]-input_width, size=num_patch-1)

            window = zip(dy, dx)

            for i in range(len(window)):
                croped = img[window[i][0]:window[i][0]+input_height, window[i][1]:window[i][1]+input_width].copy()
                #cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)
                images.append(croped)

                croped_grey = grey_img[window[i][0]:window[i][0] + input_height, window[i][1]:window[i][1] + input_width].copy()
                #lstm_image = cv2.resize(croped_grey, dsize=(lstm_sequence_length, lstm_sequence_length))
                lstm_images.append(lstm_image)

    return np.array(images), np.array(lstm_images)


def get_align_image(img_file_path):
    bgrImg = cv2.imread(img_file_path)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #print('Dlib bbox detect')
    bbs = dlibDetector(rgbImg, 1)

    if len(bbs) == 0:
        #print('No bbox')
        return [], []

    alignedFace = align.align(input_width, rgbImg, bbs[0], landmarkIndices=[8, 36, 45], skipMulti=True)

    #alignedFace = np.array(alignedFace).reshape(-1, input_width, input_height, num_channel)

    return alignedFace # RGB format


'''
def get_triplet_representation_align_image(img_file_path):
    bgrImg = cv2.imread(img_file_path)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #print('Dlib bbox detect')
    bbs = dlibDetector(rgbImg, 1)

    if len(bbs) == 0:
        #print('No bbox')
        return [], []

    alignedFace = align.align(input_width, rgbImg, bbs[0], landmarkIndices=[8, 36, 45], skipMulti=True)

    rep = triplet.forward(alignedFace)
    #alignedFace = np.array(alignedFace).reshape(-1, input_width, input_height, num_channel)

    return [rep], alignedFace # RGB format
'''


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    return gamma * loss


def make_multi_modal_noise(input, num_mode=8):
    feature = input

    for i in range(num_mode):
        noise = tf.random_normal(shape=[batch_size, 4], mean=0.0, stddev=1.0, dtype=tf.float32)
        feature = tf.concat([feature, noise], axis=1)

    return feature


def train(model_path):
    trX = []
    trY = []
    trXS = []

    teX = []
    teY = []
    teXS = []

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    for idx, labelname in enumerate(dir_list):
        imgs_list, lstm_imgs_list = load_images_from_folder(os.path.join(imgs_dirname, labelname))
        imgs_list, lstm_imgs_list = shuffle(imgs_list, lstm_imgs_list)

        label = np.zeros(len(os.listdir(imgs_dirname)))
        label[idx] += 1

        print('label:', labelname, label)

        for idx2, img in enumerate(imgs_list):
            if idx2 < len(imgs_list) * 0.8:
                trY.append(label)
                '''
                if idx2 < len(imgs_list) * 0.2:
                    # SpecAugment
                    w = np.random.randint(len(img)/10)  # Max 10% width
                    h = np.random.randint(len(img) - w + 1)
                    img[h:h + w] = [[0, 0, 0]]
                    img = np.transpose(img, [1, 0, 2])

                    w = np.random.randint(len(img)/10)  # Max 10% width
                    h = np.random.randint(len(img) - w + 1)   
                    img[h:h + w] = [[0, 0, 0]]
                    img = np.transpose(img, [1, 0, 2])

                    #cv2.imwrite(labelname + str(idx2) + '.jpg', img)
                '''
                trX.append(img)
                trXS.append(lstm_imgs_list[idx2])
            else:
                teY.append(label)
                teX.append(img)
                teXS.append(lstm_imgs_list[idx2])

    trX, trY, trXS = shuffle(trX, trY, trXS)

    trX = np.array(trX)
    trY = np.array(trY)
    trXS = np.array(trXS)
    teX = np.array(teX)
    teY = np.array(teY)
    teXS = np.array(teXS)

    trX = trX.reshape(-1, input_height, input_width, num_channel)
    teX = teX.reshape(-1, input_height, input_width, num_channel)
    trXS = trXS.reshape(-1, lstm_sequence_length, lstm_sequence_length)
    teXS = teXS.reshape(-1, lstm_sequence_length, lstm_sequence_length)

    # Network setup
    cnn_representation, _, anchor_layer = encoder_network(X, bn_phaze=bn_train, scope='encoder')
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, g_fc_layer3_dim, scope='encoder')
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    lstm_representation = lstm_network(LSTM_X, scope='lstm', forget_bias=1.0, keep_prob=keep_prob)

    # Residual
    representation = tf.concat([cnn_representation, lstm_representation], axis=1)

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, representation_dim + lstm_representation_dim],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    smoothing_factor = 0.1
    prob_threshold = 1 - smoothing_factor

    # L2 Softmax
    representation = tf.nn.l2_normalize(representation, axis=1)
    alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

    representation = tf.multiply(alpha, representation)

    center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

    with tf.device('/device:GPU:1'):
        #noise = tf.random_normal(shape=tf.shape(representation), mean=0.0, stddev=1.0, dtype=tf.float32)
        #decoder_input = tf.add(decoder_input, noise)
        decoder_input = make_multi_modal_noise(representation, num_mode=8)
        X_fake = decoder_network(decoder_input, anchor_layer=anchor_layer, activation='relu', scope='decoder', bn_phaze=bn_train)

    with tf.device('/device:GPU:2'):
        p_feature, p_logit, p_prob = discriminator(X, activation='relu', scope='discriminator', bn_phaze=bn_train)
        n_feature, n_logit, n_prob = discriminator(X_fake, activation='relu', scope='discriminator', bn_phaze=bn_train)

    # Trainable variable lists
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    lstm_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')

    generator_vars = encoder_var + decoder_var + lstm_var
    class_vars = encoder_var + lstm_var

    entropy_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=smoothing_factor))

    with tf.device('/device:GPU:1'):
        residual_loss = get_residual_loss(X, X_fake, type='l1', gamma=1.0)

    with tf.device('/device:GPU:2'):
        feature_matching_loss = get_feature_matching_loss(p_feature, n_feature, type='l2', gamma=1.0)

    # Cross Entropy
    gan_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=n_logit, labels=tf.ones_like(n_logit)))

    with tf.device('/device:GPU:2'):
        discriminator_loss, loss_real, loss_fake = get_discriminator_loss(p_logit, n_logit, type='ce', gamma=1.0)

    class_loss = entropy_loss + center_loss * LAMBDA

    # training operation
    c_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(class_loss, var_list=class_vars)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)
    gan_g_optimzier = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(gan_g_loss, var_list=generator_vars)
    f_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(feature_matching_loss,
                                                                  var_list=generator_vars)
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
            trX, trY, trXS = shuffle(trX, trY, trXS)

            for start, end in training_batch:
                sess.run(
                    [g_optimizer, residual_loss, X_fake],
                    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end],
                               bn_train: True,
                               keep_prob: 0.5})

                _, r, fake = sess.run(
                    [g_optimizer, residual_loss, X_fake],
                    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end],
                               bn_train: True,
                               keep_prob: 0.5})

                _, c, center, _ = sess.run(
                    [c_optimizer, entropy_loss, center_loss, update_center],
                    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end], bn_train: True,
                               keep_prob: 0.5})

                sample = fake[0]
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample' + str(num_itr) + '.jpg', sample)

                #_ = sess.run(
                #    [gan_g_optimzier],
                #    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end],
                #               bn_train: True,
                #               keep_prob: 0.5})

                _, d = sess.run(
                    [d_optimizer, discriminator_loss],
                    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end],
                               bn_train: True,
                               keep_prob: 0.5})

                _ = sess.run(
                    [f_optimizer],
                    feed_dict={X: trX[start:end], Y: trY[start:end], LSTM_X: trXS[start:end],
                               bn_train: True,
                               keep_prob: 0.5})

                num_itr = num_itr + 1

                if num_itr % 10 == 0:
                    print('itr #' + str(num_itr))
                    print('  - entropy loss: ' + str(c))
                    print('  - center loss: ' + str(center))
                    print('  - residual loss: ' + str(r))
                    print('  - discriminate loss: ' + str(d))

            try:
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
                           feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                      LSTM_X: teXS[test_indices], bn_train: False, keep_prob:1.0}))

            precision = np.mean(np.argmax(teY[test_indices], axis=1) ==
                             sess.run(predict_op,
                                      feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                 LSTM_X: teXS[test_indices], bn_train: False, keep_prob:1.0}))
            print('epoch ' + str(i) + ', precision: ' + str(100 * precision) + ' %')

            #if precision > 0.99:
            #    break


def int_from_bytes(b3, b2, b1, b0):
    return (((((b3 << 8) + b2) << 8) + b1) << 8) + b0


def save_unknown_user(src, dirname=None, candidate=None, confidence=0.0):
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
    #faceFile = target_dir + "/" + name + "/" + name + "_" + str(confidence) + time.strftime("_%d_%H_%M_%S_") + str(fileSeqNum) + ".jpg"
    faceFile = target_dir + "/" + name + "/" + name + "_" + str(confidence) + "_" + str(fileSeqNum) + ".jpg"

    shutil.move(src, faceFile)

    return target_dir


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def test(model_path):
    threshold = 0.9
    print('Serving Mode, threshold: ' + str(threshold))

    # Network setup
    cnn_representation, _ = encoder_network(X, bn_phaze=bn_train)
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim)
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    lstm_representation = lstm_network(LSTM_X, scope='lstm', forget_bias=1.0, keep_prob=keep_prob)

    # Residual
    representation = tf.add(cnn_representation, TripletX)
    representation = tf.concat([representation, lstm_representation], axis=1)

    smoothing_factor = 0.1
    prob_threshold = 1 - smoothing_factor

    # L2 Softmax
    representation = tf.nn.l2_normalize(representation, axis=1)
    alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

    representation = tf.multiply(alpha, representation)

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, representation_dim + lstm_representation_dim], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)
    entropy_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=0.1))

    total_loss = entropy_loss + center_loss * LAMBDA + center_distance_loss * GAMMA

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
        inputDir = fileDir + '/svm'
        label_list = [d for d in os.listdir(inputDir + '/user') if os.path.isdir(inputDir + '/user/' + d)]
        label_list.sort(key=str.lower)

        print(label_list)  # Global lable list.

        group_label_list = os.listdir(imgs_dirname)

        redis_ready = False

        clf_directory = os.path.dirname(os.path.realpath(__file__)) + '/svm/classifier/'
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
                        tpReps, img =  [], []

                        if len(tpReps) == 0:
                            print('Not a valid face.')
                        else:
                            grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            #grey_img = cv2.resize(grey_img, dsize=(lstm_sequence_length, lstm_sequence_length))
                            # grey_img = np.array(grey_img).reshape(-1, lstm_sequence_length, lstm_sequence_length)

                            #print('Run prediction..')
                            use_softmax = False

                            pred_id, confidence, rep = sess.run([predict_op, confidence_op, representation],
                                                                feed_dict={X: [img], TripletX: tpReps, LSTM_X: [grey_img], bn_train: False,
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

                                    if len(effective_labels) > 1:
                                        effective_confidences.sort(reverse=True)

                                        if effective_confidences[0] - effective_confidences[1] < 0.5:
                                            person = 'Unknown'
                                            confidence = 0.99

                                print('\nPerson: ' + person + ', Confidence: ' + str(confidence * 100) + '%')

                                if confidence < 0.9:
                                    save_unknown_user(fileName, dirname, 'Unknown', confidence)
                                elif confidence >= 0.9 and confidence < 0.97:
                                    save_unknown_user(fileName, dirname, person, confidence)

                                if confidence < threshold:
                                    person = 'Unknown'

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
    parser.add_argument('--test_data', type=str, help='test data directory', default='./test_data')
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

        if not os.path.exists(data_dir):
            print('No data.')
        else:
            # Network setup
            cnn_representation, _ = encoder_network(X, bn_phaze=bn_train)
            print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

            cnn_representation = layers.global_avg_pool(cnn_representation, g_fc_layer3_dim)
            print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

            lstm_representation = lstm_network(LSTM_X, scope='lstm', forget_bias=1.0, keep_prob=keep_prob)

            # Residual
            representation = tf.add(cnn_representation, TripletX)
            representation = tf.concat([representation, lstm_representation], axis=1)

            smoothing_factor = 0.1
            prob_threshold = 1 - smoothing_factor

            # L2 Softmax
            representation = tf.nn.l2_normalize(representation, axis=1)
            alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

            representation = tf.multiply(alpha, representation)

            prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

            with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
                centers = tf.get_variable('centers', [num_class_per_group, representation_dim + lstm_representation_dim], dtype=tf.float32,
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

                    print("image directory: " + image_directory)

                    for filename in os.listdir(image_directory):
                        fullname = os.path.join(image_directory, filename).replace("\\", "/")

                        imgs, lstm_imgs = load_images_patch(fullname)

                        for i in range(len(imgs)):
                            label_csv.writerow([idx + 1, fullname])

                            reps = sess.run(representation,
                                            feed_dict={X: [imgs[i]], LSTM_X: [lstm_imgs[i]], bn_train: False, keep_prob: 1.0})

                            reps = np.squeeze(reps).tolist()
                            embedding_csv.writerow(reps)

                        '''
                        label_csv.writerow([idx+1, fullname])

                        img = cv2.imread(fullname)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # To RGB format
                        tpReps = get_triplet_representation(img)

                        reps = sess.run(representation, feed_dict={X: [img], TripletX: [tpReps], bn_train: False, keep_prob: 1.0})

                        reps = np.squeeze(reps).tolist()
                        embedding_csv.writerow(reps)
                        '''
                label_f.close()
                embedding_f.close()
    elif mode == 'fpr':
        test_data_dir = args.test_data
        inputDir = './input'

        label_list = os.listdir(imgs_dirname)
        label_list.sort(key=str.lower)

        num_class_per_group = len(label_list)

        # Network setup
        cnn_representation, _ = encoder_network(X, bn_phaze=bn_train)
        print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

        cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim)
        print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

        lstm_representation = lstm_network(LSTM_X, scope='lstm', forget_bias=1.0, keep_prob=keep_prob)

        # Residual
        representation = tf.add(cnn_representation, TripletX)
        representation = tf.concat([representation, lstm_representation], axis=1)

        smoothing_factor = 0.1
        prob_threshold = 1 - smoothing_factor

        # L2 Softmax
        representation = tf.nn.l2_normalize(representation, axis=1)
        alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

        representation = tf.multiply(alpha, representation)

        prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final')

        with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
            centers = tf.get_variable('centers', [num_class_per_group, representation_dim + lstm_representation_dim], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0), trainable=False)

        center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
        update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)
        entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=0.1))

        total_loss = entropy_loss + center_loss * LAMBDA + center_distance_loss * GAMMA

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

            for idx, labelname in enumerate(os.listdir(test_data_dir)):
                print('label:' + labelname)

                label_dir = os.path.join(test_data_dir, labelname).replace("\\", "/")
                img_files = os.listdir(label_dir)

                for f in img_files:
                    img = []

                    if len(img) == 0 :
                        print('No valid face')
                    else:
                        #img = np.array(img).reshape(input_width, input_height, num_channel)
                        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        #grey_img = cv2.resize(grey_img, dsize=(lstm_sequence_length, lstm_sequence_length))
                        grey_img = np.array(grey_img).reshape(-1, lstm_sequence_length, lstm_sequence_length)

                        pred_id, confidence = sess.run([predict_op, confidence_op],
                                                       feed_dict={X: [img],
                                                                  LSTM_X: grey_img, bn_train: False, keep_prob: 1.0})

                        #idx = pred_id[0]
                        #center = c[idx]
                        #cos = findCosineDistance(center, rep[0]) * 100

                        print(
                            labelname + ', predict: ' + label_list[pred_id[0]] + ', ' + str(confidence[0][pred_id[0]]))
