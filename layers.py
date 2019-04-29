import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import util
import argparse
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_feature_dim = 97
cond_step_dim = 8
cond_wafer_dim = 24
cond_dim = cond_step_dim + cond_wafer_dim

lstm_sequence_length = 20
lstm_hidden_size_layer1 = 64
lstm_hidden_size_layer2 = 64
lstm_feature_dim = lstm_hidden_size_layer1
lstm_z_sequence_dim = 16
lstm_linear_transform_input_dim = 2 * lstm_feature_dim

g_encoder_z_local_dim = 16
g_encoder_z_dim = lstm_z_sequence_dim + g_encoder_z_local_dim + cond_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 84
g_encoder_layer2_dim = 64
g_encoder_layer3_dim = 32

g_decoder_output_dim = input_feature_dim
g_decoder_layer2_dim = 72
g_decoder_layer1_dim = 84

d_layer_1_dim = input_feature_dim
d_layer_2_dim = 64
d_layer_3_dim = 32
d_layer_4_dim = 16

num_block_layers = 3
dense_layer_depth = 16


def lstm_network(input, scope='lstm_network'):
    with tf.variable_scope(scope):
        # tf.nn.rnn_cell
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # tf.nn.rnn_cell
        # lstm_cell1 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        # lstm_cell2 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        #lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # initial_state = lstm_cells.zero_state(batch_size,  tf.float32)

        _, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)

        # z_sequence_output = states[1].h
        # print(z_sequence_output.get_shape())
        states_concat = tf.concat([states[0].h, states[1].h], 1)

        #def fc(input, scope, out_dim, non_linear_fn=None, initial_value=None, use_bias=True):
        z_sequence_output = fc(states_concat, lstm_z_sequence_dim, scope='linear_transform')

    return z_sequence_output


def fc(input_data, out_dim, non_linear_fn=None, initial_value=None, use_bias=True, scope='fc'):
    with tf.variable_scope(scope):
        input_dims = input_data.get_shape().as_list()

        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_data, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_data

        if initial_value is None:
            fc_weight = tf.get_variable("weights", shape=[in_dim, out_dim], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=tf.constant_initializer(0.0))
        else:
            fc_weight = tf.get_variable("weights", initializer=initial_value[0])
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=initial_value[1])

        if use_bias:
            output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)
        else:
            output = tf.matmul(flat_input, fc_weight)

        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def batch_norm(x, b_train, scope, reuse=False):
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, bias=True):
    input_dims = input.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        conv_weight = tf.Variable(
            tf.truncated_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))
        conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))
        map = tf.nn.conv2d(input, conv_weight, strides=[1, stride_h, stride_w, 1], padding=padding)

        if bias is True:
            map = tf.nn.bias_add(map, conv_bias)

        if non_linear_fn is not None:
            activation = non_linear_fn(map)
        else:
            activation = map

        # print(activation.get_shape().as_list())
        return activation


def batch_norm_conv(x, b_train, scope):
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def add_dense_layer(layer, filter_dims, act_func=tf.nn.relu, scope='dense_layer', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = layer

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = act_func(l)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=None, bias=False)
        l = tf.concat([l, layer], 3)

    return l


def add_dense_transition_layer(layer, filter_dims, act_func=tf.nn.relu, scope='transition', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = act_func(layer)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=None, bias=False)

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        #def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
        l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return l


def global_avg_pool(input_data, output_length=1, padding='VALID', scope='gloval_avg_pool'):
    input_dims = input_data.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in

    num_channels_in = input_dims[-1]
    height = input_dims[1]
    width = input_dims[2]

    with tf.variable_scope(scope):
        if output_length == 1:
            pool = tf.nn.avg_pool(input_data, [1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
            pool = tf.reduce_mean(pool, axis=[1, 2])
            pool = tf.squeeze(pool, axis=[1, 2])

            return pool
        else:
            conv_weight = tf.Variable(tf.truncated_normal([1, 1, num_channels_in, output_length], stddev=0.1, dtype=tf.float32))
            conv = tf.nn.conv2d(input_data, conv_weight, strides=[1, 1, 1, 1], padding='SAME')
            pool = tf.nn.avg_pool(conv, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
            pool = tf.squeeze(pool, axis=[1, 2])

            return pool


def avg_pool(input, scope, filter_dims, stride_dims, padding='SAME'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.avg_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool
