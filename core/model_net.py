#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/08 22:26
# @Author   : WanDaoYi
# @FileName : model_net.py
# ============================================

import tensorflow as tf
from core.common import Common


class ModelNet(object):

    def __init__(self):
        self.common = Common()
        pass

    def lenet_5(self, input_data, n_label=10, keep_prob=1.0, train_flag=True):
        with tf.variable_scope("lenet-5"):
            conv_1 = self.common.conv2d(input_data, (5, 5, 1, 6), name="conv_1")
            tanh_1 = tf.nn.tanh(conv_1, name="tanh_1")
            avg_pool_1 = self.common.deal_pool(tanh_1, name="avg_pool_1")

            conv_2 = self.common.conv2d(avg_pool_1, (5, 5, 6, 16), name="conv_2")
            tanh_2 = tf.nn.tanh(conv_2, name="tanh_2")
            avg_pool_2 = self.common.deal_pool(tanh_2, name="avg_pool_2")

            conv_3 = self.common.conv2d(avg_pool_2, (5, 5, 16, 120), name="conv_3")
            tanh_3 = tf.nn.tanh(conv_3, name="tanh_3")

            reshape_data = tf.reshape(tanh_3, [-1, 120])

            dropout_1 = self.common.deal_dropout(reshape_data, keep_prob)

            fc_1 = self.common.neural_layer(dropout_1, 84, name="fc_1")
            tanh_4 = tf.nn.tanh(fc_1, name="tanh_4")

            dropout_2 = self.common.deal_dropout(tanh_4, keep_prob)

            fc_2 = self.common.neural_layer(dropout_2, n_label, name="fc_2")
            scale_2 = self.common.deal_bn(fc_2, train_flag=train_flag)
            result_info = tf.nn.softmax(scale_2, name="result_info")

            return result_info

        pass



