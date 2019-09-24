import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
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
import alignment

LAMBDA = 1e-2
GAMMA = 1.0

CENTER_LOSS_ALPHA = 1.0
DISTANCE_MARGIN = 10.0

representation_dim = 256
input_width = 96
input_height = 96
scale_size = 112
num_channel = 3
num_patch = 4
batch_size = 16
test_size = 100
num_class_per_group = 70
num_epoch = 30

# Network Parameters
g_fc_layer1_dim = 1024
g_fc_layer2_dim = 512  # Final representation
g_fc_layer3_dim = 128

g_dense_block_layers = 4
g_dense_block_depth = 128

lstm_hidden_size_layer1 = 128
lstm_hidden_size_layer2 = 128
lstm_sequence_length = 96
lstm_representation_dim = 64

dlibDetector = dlib.get_frontal_face_detector()
align = alignment.AlignDlib('dlib/shape_predictor_68_face_landmarks.dat')

with tf.device('/device:CPU:0'):
    ANCHOR = tf.placeholder(tf.float32, [None, 48, 48, 128])

#LSTM_X = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, lstm_sequence_length])
bn_train = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def get_center_loss(features, labels):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', initializer=tf.random_normal_initializer)

    len_features = features.get_shape()[1]

    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.reduce_sum((features - centers_batch) ** 2, [1]))

    unique_labels, _ = tf.unique(labels)
    reverse_labels = tf.reverse(unique_labels, [0])

    # Center distance loss
    unique_centers = tf.gather(centers, unique_labels)
    shuffle_centers = tf.gather(centers, reverse_labels)

    distance = tf.reduce_mean(tf.reduce_sum((unique_centers - shuffle_centers) ** 2, [1]))

    distance_loss = DISTANCE_MARGIN / (1 + distance)
    #distance_loss = DISTANCE_MARGIN * (1.0 - tf.sigmoid(distance))

    return loss * distance_loss, distance_loss


def update_centers(features, labels, alpha):
    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', initializer=tf.random_normal_initializer)

    labels = tf.reshape(labels, [-1])  # flatten
    centers_batch = tf.gather(centers, labels)  # Gather center tensor by labels value order
    diff = centers_batch - features  # L1 distance array between each of center and feature

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers = tf.scatter_sub(centers, labels, diff)

    return centers


def fc_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='fc_network', bn_phaze=False,
               keep_prob=0.5):
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
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_representation_dim / 2)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_representation_dim / 2)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

        _, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)

        # print(states)
        # print(states[0])
        # print(states[1])

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
        l = tf.reshape(x, shape=[-1, 4, 4, 12])

        # 6 x 6
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv1', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        print('deconv1:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_0')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn')
        l = act_func(l)

        # 12 x 12
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv2', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)
        print('deconv2:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_1')
        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn')
        l = act_func(l)

        # 24 x 24
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv3', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)
        print('deconv3:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_2')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn')
        l = act_func(l)
        # 48 x 48
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv4', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        print('deconv4:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_3')

        # l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
        #                       act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_3_1')

        if anchor_layer is not None:
            # anchor_layer = anchor_layer + tf.random_normal(shape=tf.shape(anchor_layer), mean=0.0, stddev=1.0, dtype=tf.float32)
            l = tf.concat([l, anchor_layer], axis=3)

        # l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn')
        l = act_func(l)
        # 96 x 96
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_conv5', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=act_func)

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_4')

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_4_1')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
        #                       act_func=act_func, bn_phaze=bn_phaze, scope='block_4_1')

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, 3], act_func=act_func,
                                              scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        # l = act_func(l)

        print('deconv5:', str(l.get_shape().as_list()))

        return l


def discriminator(input_data, activation='swish', scope='discriminator', reuse=False, bn_phaze=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # if reuse:
        #    tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'tanh':
            act_func = tf.nn.tanh
        else:
            act_func = tf.nn.sigmoid

        l = layers.conv(input_data, scope='conv1', filter_dims=[3, 3, g_dense_block_depth / 2], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth / 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_0')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth / 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth / 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        # l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth/2], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        # l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth/2], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth/2], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_5')

        # l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_6')

        # l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_7')

        # dc_final_layer = batch_norm_conv(last_dense_layer, b_train=bn_phaze, scope='last_dense_layer')

        l = layers.global_avg_pool(l, representation_dim)
        dc_final_layer = l

        dc_output = layers.fc(dc_final_layer, scope='g_enc_z_fc', out_dim=1, non_linear_fn=None)

    return dc_final_layer, dc_output, tf.sigmoid(dc_output)


def add_residual_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, bn_phaze=False, use_residual=True,
                       scope='residual_block'):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = input_dims[-1]

        for i in range(num_layers):
            l = layers.add_residual_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                          scope='layer' + str(i))

        if use_residual is True:
            l = tf.add(l, in_layer)

    return l


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, bn_phaze=False,
                             scope='residual_dense_block', use_dilation=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        for i in range(num_layers):
            l = layers.add_dense_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i), dilation=dilation)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_out], act_func=act_func,
                                              scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        l = tf.add(l, in_layer)

    return l


def encoder_network(x, activation='relu', scope='encoder_network', reuse=False, bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # if reuse:
        #    tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        # [96 x 96]
        l = layers.conv(x, scope='conv1', filter_dims=[3, 3, g_dense_block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_0')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn1')
        l = act_func(l)

        # [48 x 48]
        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3_1')

        l = layers.self_attention(l, g_dense_block_depth)

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn2')
        l = act_func(l)

        #l = layers.self_attention(l, g_dense_block_depth, act_func=act_func)

        l_share = l

        # [24 x 24]
        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth*2],
                                          act_func=act_func,
                                          scope='dense_transition_24', bn_phaze=bn_phaze,
                                          use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn3')
        l = act_func(l)

        # [12 x 12]
        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth * 3],
                                              act_func=act_func,
                                              scope='dense_transition_12', bn_phaze=bn_phaze,
                                              use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_6')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn4')
        l = act_func(l)

        # [6 x 6]
        l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth * 4],
                                              act_func=act_func,
                                              scope='dense_transition_6', bn_phaze=bn_phaze,
                                              use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*4], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_8')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*4], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_9')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth*4], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_10')

        with tf.variable_scope('dense_block_last'):
            scale_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim],
                                                            act_func=act_func,
                                                            scope='dense_transition_1', bn_phaze=bn_phaze,
                                                            use_pool=False)
            last_dense_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim],
                                                                 act_func=act_func,
                                                                 scope='dense_transition_2', bn_phaze=bn_phaze,
                                                                 use_pool=False)
            scale_layer = act_func(scale_layer)
            last_dense_layer = act_func(last_dense_layer)

    return last_dense_layer, scale_layer, l_share


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

        dy = np.random.random_integers(low=1, high=img.shape[0] - input_height, size=num_patch - 1)
        dx = np.random.random_integers(low=1, high=img.shape[1] - input_width, size=num_patch - 1)

        window = zip(dy, dx)

        for i in range(len(window)):
            croped = img[window[i][0]:window[i][0] + input_height, window[i][1]:window[i][1] + input_width].copy()
            # cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)
            images.append(croped)

            croped_grey = grey_img[window[i][0]:window[i][0] + input_height,
                          window[i][1]:window[i][1] + input_width].copy()
            lstm_image = croped_grey / 255.0
            lstm_images.append(lstm_image)

    images = np.array(images)
    lstm_images = np.array(lstm_images)

    return images, lstm_images


def load_images_from_folder(folder, use_augmentation=False):
    images = []
    lstm_images = []

    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        grey_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)

        if img is not None:
            img = np.array(img)
            grey_img = np.array(grey_img)

            n_img = img / 255.0
            images.append(n_img)

            n_img = cv2.flip(img, 1)
            n_img = n_img / 255.0
            images.append(n_img)

            lstm_image = cv2.resize(grey_img, dsize=(lstm_sequence_length, lstm_sequence_length))
            lstm_image = lstm_image / 255.0
            lstm_images.append(lstm_image)

            lstm_image = cv2.flip(grey_img, 1)
            lstm_image = lstm_image / 255.0
            lstm_images.append(lstm_image)

            # scale = scale_size / input_width

            if use_augmentation == True:
                img = cv2.resize(img, dsize=(scale_size, scale_size), interpolation=cv2.INTER_CUBIC)
                grey_img = cv2.resize(grey_img, dsize=(scale_size, scale_size), interpolation=cv2.INTER_CUBIC)

                dy = np.random.random_integers(low=1, high=img.shape[0] - input_height, size=num_patch - 1)
                dx = np.random.random_integers(low=1, high=img.shape[1] - input_width, size=num_patch - 1)

                window = zip(dy, dx)

                for i in range(len(window)):
                    croped = img[window[i][0]:window[i][0] + input_height, window[i][1]:window[i][1] + input_width].copy()
                    # cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)
                    n_croped = croped / 255.0
                    images.append(n_croped)

                    croped = cv2.flip(croped, 1)
                    croped = croped / 255.0
                    images.append(croped)

                    croped_grey = grey_img[window[i][0]:window[i][0] + input_height,
                                  window[i][1]:window[i][1] + input_width].copy()

                    lstm_image = croped_grey / 255.0
                    lstm_images.append(lstm_image)

                    croped_grey = cv2.flip(croped_grey, 1)
                    lstm_image = croped_grey / 255.0
                    lstm_images.append(lstm_image)

    return np.array(images), np.array(lstm_images)


def get_align_image(img_file_path):
    bgrImg = cv2.imread(img_file_path)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    # print('Dlib bbox detect')
    bbs = dlibDetector(rgbImg, 1)

    if len(bbs) == 0:
        # print('No bbox')
        return [], []

    alignedFace = align.align(input_width, rgbImg, bbs[0], landmarkIndices=alignment.AlignDlib.INNER_EYES_AND_NOSE, skipMulti=True)

    return alignedFace  # RGB format


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

    loss = gamma * loss

    return loss


def make_multi_modal_noise(input, num_mode=8):
    feature = input

    for i in range(num_mode):
        noise = tf.random_normal(shape=[batch_size, 8], mean=0.0, stddev=2.0, dtype=tf.float32)
        feature = tf.concat([feature, noise], axis=1)

    return feature


def train(model_path):
    trX = []
    trY = []

    teX = []
    teY = []

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    one_hot_length = len(os.listdir(imgs_dirname))

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        Y = tf.placeholder(tf.float32, [None, num_class_per_group])

        for idx, labelname in enumerate(dir_list):
            imgs_list, _ = load_images_from_folder(os.path.join(imgs_dirname, labelname), use_augmentation=True)
            imgs_list = shuffle(imgs_list)

            label = np.zeros(one_hot_length)
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

                        lstm_imgs_list[idx2][h:h+w] = [[0]]
                        #cv2.imwrite(labelname + str(idx2) + '.jpg', img)
                    '''
                    trX.append(img)
                else:
                    if labelname != 'Unknown':
                        teY.append(label)
                        teX.append(img)

        trX, trY = shuffle(trX, trY)

        trX = np.array(trX)
        trY = np.array(trY)
        teX = np.array(teX)
        teY = np.array(teY)

        trX = trX.reshape(-1, input_height, input_width, num_channel)
        teX = teX.reshape(-1, input_height, input_width, num_channel)

    print('Number of Classes: ' + str(num_class_per_group))
    # Network setup
    cnn_representation, _, anchor_layer = encoder_network(X, bn_phaze=bn_train, activation='relu', scope='encoder')
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim, scope='encoder')
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    representation = cnn_representation

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, representation_dim],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer, trainable=False)

    smoothing_factor = 0.1
    prob_threshold = 1 - smoothing_factor

    # L2 Softmax
    # representation = tf.nn.l2_normalize(cnn_representation, axis=1)
    # alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

    # representation = tf.multiply(alpha, representation)

    # [Batch, representation_dim]
    # representation = tf.nn.l2_normalize(representation, axis=1)

    center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final_2')

    if smoothing_factor > 0:
        entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=smoothing_factor))
    else:
        entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction))

    class_loss = entropy_loss + center_loss * LAMBDA

    # training operation
    c_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(class_loss)
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
            print('Start New Training. Wait ...')

        num_itr = 0
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))

        for i in range(num_epoch):
            trX, trY = shuffle(trX, trY)

            for start, end in training_batch:

                _, c, center, cd, _ = sess.run(
                    [c_optimizer, entropy_loss, center_loss, center_distance_loss, update_center],
                    feed_dict={X: trX[start:end], Y: trY[start:end], bn_train: True,
                               keep_prob: 0.5})

                num_itr = num_itr + 1

                if num_itr % 10 == 0:
                    print('itr #' + str(num_itr))
                    print('  - entropy loss: ' + str(c))
                    print('  - center loss: ' + str(center))
                    print('  - distance loss: ' + str(cd))

            try:
                saver.save(sess, model_path)
            except:
                print('Save failed')

            test_indices = np.arange(len(teX))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:100]

            print('# Test Set #')
            print(np.argmax(teY[test_indices], axis=1))

            print('# Prediction #')
            print(sess.run(predict_op,
                           feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                      bn_train: False, keep_prob: 1.0}))

            precision = np.mean(np.argmax(teY[test_indices], axis=1) ==
                                sess.run(predict_op,
                                         feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                    bn_train: False, keep_prob: 1.0}))
            print('epoch ' + str(i) + ', precision: ' + str(100 * precision) + ' %')


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
    # faceFile = target_dir + "/" + name + "/" + name + "_" + str(confidence) + time.strftime("_%d_%H_%M_%S_") + str(fileSeqNum) + ".jpg"
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

    X = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    Y = tf.placeholder(tf.float32, [None, num_class_per_group])

    print('Number of Classes: ' + str(num_class_per_group))
    # Network setup
    cnn_representation, _, anchor_layer = encoder_network(X, bn_phaze=bn_train, activation='relu', scope='encoder')
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim, scope='encoder')
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    representation = cnn_representation

    with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
        centers = tf.get_variable('centers', [num_class_per_group, representation_dim],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer, trainable=False)

    smoothing_factor = 0.1
    prob_threshold = 1 - smoothing_factor

    # L2 Softmax
    # representation = tf.nn.l2_normalize(cnn_representation, axis=1)
    # alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

    # representation = tf.multiply(alpha, representation)

    # [Batch, representation_dim]
    #representation = tf.nn.l2_normalize(representation, axis=1)

    center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
    update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)

    prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final_2')

    if smoothing_factor > 0:
        entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction, label_smoothing=smoothing_factor))
    else:
        entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=prediction))

    # class_loss = entropy_loss + center_loss * LAMBDA
    center_full_loss = CENTER_LOSS_ALPHA * center_loss + center_distance_loss
    class_loss = entropy_loss + center_full_loss * LAMBDA

    # training operation
    c_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(class_loss)
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

        # Modify baseDir to your environment
        label_list = [d for d in os.listdir(label_directory) if os.path.isdir(label_directory + '/' + d)]
        label_list.sort(key=str.lower)

        print(label_list)  # Global label list.

        group_label_list = os.listdir(imgs_dirname)
        group_label_list.sort(key=str.lower)

        redis_ready = False

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

        dirname = './Unknown'

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

                        img = get_align_image(fileName)

                        if len(img[0]) == 0:
                            print('Not a valid face.')
                        else:
                            img = img / 255.0

                            pred_id, confidence, rep = sess.run([predict_op, confidence_op, representation],
                                                                feed_dict={X: [img], bn_train: False, keep_prob: 1.0})

                            # print('# Prediction: ' + str(pred_id))
                            person = group_label_list[pred_id[0]]
                            confidence = confidence[0][pred_id[0]]

                            print('# Person: ' + person + ', Confidence: ' + str(confidence))

                            if confidence < threshold:
                                save_unknown_user(fileName, dirname, 'Unknown', confidence)
                            elif confidence >= threshold:
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
        print('Num classes: ' + str(num_class_per_group))
        train(model_path)
    elif mode == 'test':
        HOST = args.camera_host
        PORT = args.camera_port
        REDIS_SERVER = args.redis_host
        REDIS_PORT = args.redis_port
        redis_channel = args.redis_camera_channel
        frame_db = args.redis_frame_channel

        num_class_per_group = len(os.listdir(imgs_dirname))

        test(model_path)
    elif mode == 'fpr':
        test_data_dir = args.test_data
        inputDir = './input'

        label_list = os.listdir(imgs_dirname)
        label_list.sort(key=str.lower)

        num_class_per_group = len(label_list)

        X = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        Y = tf.placeholder(tf.float32, [None, num_class_per_group])

        print('Number of Classes: ' + str(num_class_per_group))
        # Network setup
        cnn_representation, _, anchor_layer = encoder_network(X, bn_phaze=bn_train, activation='relu', scope='encoder')
        print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

        cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim, scope='encoder')
        print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

        representation = cnn_representation

        with tf.variable_scope('center', reuse=tf.AUTO_REUSE):
            centers = tf.get_variable('centers', [num_class_per_group, representation_dim],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer, trainable=False)

        smoothing_factor = 0.1
        prob_threshold = 1 - smoothing_factor

        # L2 Softmax
        # representation = tf.nn.l2_normalize(cnn_representation, axis=1)
        # alpha = tf.log((prob_threshold * (num_class_per_group - 2)) / smoothing_factor)

        # representation = tf.multiply(alpha, representation)

        # [Batch, representation_dim]
        #representation = tf.nn.l2_normalize(representation, axis=1)

        center_loss, center_distance_loss = get_center_loss(representation, tf.argmax(Y, 1))
        update_center = update_centers(representation, tf.argmax(Y, 1), CENTER_LOSS_ALPHA)

        prediction = layers.fc(representation, num_class_per_group, scope='g_fc_final_2')

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

            b_align = False

            for idx, labelname in enumerate(os.listdir(test_data_dir)):

                if os.path.isdir(os.path.join(test_data_dir, labelname).replace("\\", "/")) is False:
                    continue

                print('label:' + labelname)

                label_dir = os.path.join(test_data_dir, labelname).replace("\\", "/")
                img_files = os.listdir(label_dir)

                for f in img_files:
                    if b_align == True:
                        img = get_align_image(os.path.join(label_dir, f).replace("\\", "/"))
                    else:
                        bgrImg = cv2.imread(os.path.join(label_dir, f).replace("\\", "/"))
                        img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

                    if len(img[0]) == 0:
                        print('No valid face')
                    else:
                        img = img / 255.0

                        pred_id, confidence = sess.run([predict_op, confidence_op],
                                                       feed_dict={X: [img], bn_train: False, keep_prob: 1.0})

                        # idx = pred_id[0]
                        # center = c[idx]
                        # cos = findCosineDistance(center, rep[0]) * 100

                        print(
                            labelname + ', predict: ' + label_list[pred_id[0]] + ', ' + str(confidence[0][pred_id[0]]))

                        #print(l)
