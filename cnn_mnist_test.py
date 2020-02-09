#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/08 19:24
# @Author   : WanDaoYi
# @FileName : cnn_mnist_test.py
# ============================================

from datetime import datetime
import tensorflow as tf
import numpy as np
from config import cfg
from core.common import Common
from core.model_net import ModelNet


class CnnMnistTest(object):

    def __init__(self):
        self.common = Common()
        self.model_net = ModelNet()
        # 读取数据和 维度
        self.mnist_data, self.n_feature, self.n_label = self.common.read_data()

        # ckpt 模型
        self.test_ckpt_model = cfg.TEST.CKPT_MODEL_SAVE_PATH
        print("test_ckpt_model: {}".format(self.test_ckpt_model))

        # tf.reset_default_graph()
        # 创建设计图
        with tf.name_scope(name="input"):
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.n_feature), name="input_data")
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_label), name="input_labels")

        self.data_shape = cfg.COMMON.DATA_RESHAPE
        self.data_resize = cfg.COMMON.DATA_RESIZE
        with tf.name_scope(name="input_shape"):
            # 784维度变形为图片保持到节点
            # -1 代表进来的图片的数量、28 x 28 是图片的高和宽，1是图片的颜色通道
            self.image_shaped_input = tf.reshape(self.x, self.data_shape)
            # 将 输入 图像 resize 成 网络所需要的大小 32 x 32
            self.image_resize = tf.image.resize_images(self.image_shaped_input, self.data_resize)

        # 获取最后一层 lenet_5 的返回结果
        self.result_info = self.model_net.lenet_5(self.image_resize, n_label=self.n_label)

        pass

    # 预测
    def do_ckpt_test(self):

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.test_ckpt_model)

            # 预测
            output = self.result_info.eval(feed_dict={self.x: self.mnist_data.test.images})

            # 将 one-hot 预测值转为 数字
            y_perd = np.argmax(output, axis=1)
            print("预测值: {}".format(y_perd[: 5]))

            # 真实值
            y_true = np.argmax(self.mnist_data.test.labels, axis=1)
            print("真实值: {}".format(y_true[: 5]))
            pass

        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = CnnMnistTest()
    # 使用 ckpt 模型测试
    demo.do_ckpt_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))

