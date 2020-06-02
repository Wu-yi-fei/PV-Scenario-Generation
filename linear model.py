"""
tensorflow做线性回归模型（测试项目）
"""

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33  # 创建一些线性值附近的随机值

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    return tf.multiply(X, w)  # X*w线性求值


w = tf.Variable(0.0, name="weights")
y_model = model(X, w)

cost = tf.square(Y - y_model)  # 用平方误差做为优化目标

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # 梯度下降优化器

# 开始创建Session干活
with tf.Session() as sess:
    # 首先需要初始化全局变量，这是Tensorflow的要求
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))
