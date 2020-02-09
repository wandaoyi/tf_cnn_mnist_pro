#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/08 19:24
# @Author   : WanDaoYi
# @FileName : cnn_mnist_train.py
# ============================================

from datetime import datetime
import tensorflow as tf
from config import cfg
from core.common import Common
from core.model_net import ModelNet


class CnnMnistTrain(object):

    def __init__(self):
        # 模型保存路径
        self.model_save_path = cfg.TRAIN.MODEL_SAVE_PATH
        self.log_path = cfg.LOG.LOG_SAVE_PATH

        self.learning_rate = cfg.TRAIN.LEARNING_RATE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.n_epoch = cfg.TRAIN.N_EPOCH

        self.data_shape = cfg.COMMON.DATA_RESHAPE
        self.data_resize = cfg.COMMON.DATA_RESIZE

        self.common = Common()
        self.model_net = ModelNet()
        # 读取数据和 维度
        self.mnist_data, self.n_feature, self.n_label = self.common.read_data()

        # 创建设计图
        with tf.name_scope(name="input_data"):
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.n_feature), name="input_data")
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_label), name="input_labels")

        with tf.name_scope(name="input_shape"):
            # 784维度变形为图片保持到节点
            # -1 代表进来的图片的数量、28，28是图片的高和宽，1是图片的颜色通道
            image_shaped_input = tf.reshape(self.x, self.data_shape)
            # 将 输入 图像 resize 成 网络所需要的大小
            image_resize = tf.image.resize_images(image_shaped_input, self.data_resize)
            tf.summary.image('input', image_resize, self.n_label)

        self.keep_prob_dropout = cfg.TRAIN.KEEP_PROB_DROPOUT
        self.keep_prob = tf.placeholder(tf.float32)

        # 获取最后一层 lenet_5 的返回结果
        self.result_info = self.model_net.lenet_5(image_resize, n_label=self.n_label,
                                                  keep_prob=self.keep_prob_dropout)

        # 计算损失
        with tf.name_scope(name="train_loss"):
            # 定义损失函数
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.result_info),
                                                               reduction_indices=[1]))
            tf.summary.scalar("train_loss", self.cross_entropy)
            pass

        with tf.name_scope(name="optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.cross_entropy)
            pass

        with tf.name_scope(name="accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.result_info, 1), tf.argmax(self.y, 1))
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            tf.summary.scalar("accuracy", self.acc)
            pass

        # 因为我们之前定义了太多的tf.summary汇总操作，逐一执行这些操作太麻烦，
        # 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
        self.merged = tf.summary.merge_all()

        self.sess = tf.InteractiveSession()
        # 保存训练模型
        self.saver = tf.train.Saver()

        # 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
        # 同时，将Session计算图sess.graph加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
        self.train_writer = tf.summary.FileWriter(self.log_path + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_path + 'test')

        pass

    # 灌入数据
    def feed_dict(self, train_flag=True):
        # 训练样本
        if train_flag:
            # 获取下一批次样本
            x_data, y_data = self.mnist_data.train.next_batch(self.batch_size)
            keep_prob = self.keep_prob_dropout
            pass
        # 验证样本
        else:
            x_data, y_data = self.mnist_data.test.images, self.mnist_data.test.labels
            keep_prob = 1.0
            pass
        return {self.x: x_data, self.y: y_data, self.keep_prob: keep_prob}
        pass

    def do_train(self):
        # 定义初始化
        init = tf.global_variables_initializer()
        self.sess.run(init)

        test_acc = None
        for epoch in range(self.n_epoch):
            # 获取总样本数量
            batch_number = self.mnist_data.train.num_examples
            # 获取总样本一共几个批次
            size_number = int(batch_number / self.batch_size)
            for number in range(size_number):
                summary, _ = self.sess.run([self.merged, self.train_op], feed_dict=self.feed_dict())

                # 第几次循环
                i = epoch * size_number + number + 1
                self.train_writer.add_summary(summary, i)

                if number == size_number - 1:
                    # 获取下一批次样本
                    x_batch, y_batch = self.mnist_data.train.next_batch(self.batch_size)
                    acc_train = self.acc.eval(feed_dict={self.x: x_batch, self.y: y_batch})
                    print("acc_train: {}".format(acc_train))

            # 验证 方法二 两个方法，随便挑一个都可以的。
            test_summary, acc_test = self.sess.run([self.merged, self.acc], feed_dict=self.feed_dict(False))
            print("epoch: {}, acc_test: {}".format(epoch + 1, acc_test))
            self.test_writer.add_summary(test_summary, epoch + 1)

            test_acc = acc_test
            pass

        save_path = self.model_save_path + "acc={:.6f}".format(test_acc) + ".ckpt"
        # 保存模型
        self.saver.save(self.sess, save_path, global_step=self.n_epoch)

        self.train_writer.close()
        self.test_writer.close()

        pass


if __name__ == "__main__":

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = CnnMnistTrain()
    demo.do_train()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
