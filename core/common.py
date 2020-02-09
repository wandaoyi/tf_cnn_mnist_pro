#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/08 19:26
# @Author   : WanDaoYi
# @FileName : common.py
# ============================================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from config import cfg
import numpy as np


class Common(object):

    def __init__(self):
        # 数据路径
        self.data_file_path = cfg.COMMON.DATA_PATH

        pass

    # 读取数据
    def read_data(self):
        # 数据下载地址: http://yann.lecun.com/exdb/mnist/
        mnist_data = input_data.read_data_sets(self.data_file_path, one_hot=True)
        train_image = mnist_data.train.images
        train_label = mnist_data.train.labels
        _, n_feature = train_image.shape
        _, n_label = train_label.shape

        return mnist_data, n_feature, n_label

    # bn 操作
    def deal_bn(self, input_data, train_flag=True):
        bn_info = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(),
                                                training=train_flag)
        return bn_info
        pass

    # 池化处理
    def deal_pool(self, input_data, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                  padding="VALID", name="avg_pool"):
        pool_info = tf.nn.avg_pool(value=input_data, ksize=ksize,
                                   strides=strides, padding=padding,
                                   name=name)
        tf.summary.histogram('pooling', pool_info)
        return pool_info
        pass

    # dropout 处理
    def deal_dropout(self, hidden_layer, keep_prob):
        with tf.name_scope("dropout"):
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden_layer, keep_prob)
            tf.summary.histogram('dropped', dropped)
            return dropped
        pass

    # 参数记录
    def variable_summaries(self, param):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(param)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(param))
            tf.summary.scalar('min', tf.reduce_min(param))
            tf.summary.histogram('histogram', param)

    # 全连接操作
    def neural_layer(self, x, n_neuron, name="fc"):
        # 包含所有的计算节点对于这一层, name_scope 可写可不写
        with tf.name_scope(name=name):
            n_input = int(x.get_shape()[1])
            stddev = 2 / np.sqrt(n_input)

            # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
            # truncated normal distribution 比 regular normal distribution的值小
            # 不会出现任何大的权重值，确保慢慢的稳健的训练
            # 使用这种标准方差会让收敛快
            # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
            with tf.name_scope("weights"):
                init_w = tf.truncated_normal((n_input, n_neuron), stddev=stddev)
                w = tf.Variable(init_w, name="weight")
                self.variable_summaries(w)

            with tf.name_scope("biases"):
                b = tf.Variable(tf.zeros([n_neuron]), name="bias")
                self.variable_summaries(b)
            with tf.name_scope("wx_plus_b"):
                z = tf.matmul(x, w) + b
                tf.summary.histogram('pre_activations', z)

            return z

    # 卷积操作
    def conv2d(self, input_data, filter_shape, strides_shape=(1, 1, 1, 1),
               padding="VALID", train_flag=True, name="conv2d"):
        with tf.variable_scope(name):
            weight = tf.get_variable(name="weight", dtype=tf.float32,
                                     trainable=train_flag,
                                     shape=filter_shape,
                                     initializer=tf.random_normal_initializer(stddev=0.01))

            conv = tf.nn.conv2d(input=input_data, filter=weight,
                                strides=strides_shape, padding=padding)

            conv_2_bn = self.deal_bn(conv, train_flag=train_flag)

            return conv_2_bn
            pass
        pass
