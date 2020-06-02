"""
光伏发电GAN模型场景预测
这种方法是无模型和数据的，它生成一组场景，这些场景表示基于历史观察和点预测的未来可能的行为。
通过这种方法，我们完成了场景的定向预测，给生成的场景“指明”了“方向”。
"""

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
# import scipy.misc
from numpy import shape
import csv
import numpy as np
import os

sys.path.append('..')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
一些辅助函数:
包括一些编码函数和存储函数
"""


def OneHot(X, n, negative_class=0.):
    X = np.asarray(X).flatten()  # 扁平化->数组降维
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class  # 定义了一个shape为(len(X),n)的编码数组
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


"""
数据准备：
加载光伏历史数据(csv类型文件)以及日前预测到模型中，并重新塑成可调的模型输入形状
Label只对基于事件的场景生成有用
"""


def load_solar_data_new():
    # 为基于事件的光伏场景生成进一步创建和处理合适的数据集，
    # 数据来自于NREL的集成数据集，观测地点是加利福尼亚州。
    with open('datasets/solar_datasets.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of solar", m)
    print(shape(rows))
    for x in range(52):
        train = rows[:-288, x].reshape(-1, 576)
        train = train / 8

        print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape TrX", shape(trX))

    with open('datasets/solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    trY = np.array(rows, dtype=int)
    trY = np.tile(trY, (52, 1))  # 延展至9464
    print("Label shape----", shape(trY))

    with open('predict dataset/index.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = [row for row in reader]
    index = np.array(index, dtype=int)

    print(shape(index))

    trX2 = trX[index[0:8464]]
    trY2 = trY[index[0:8464]]
    trX2 = trX2.reshape([-1, 576])
    teX = trX[index[8464:]]
    teX = teX.reshape([-1, 576])
    teY = trY[index[8464:]]

    with open('predict dataset/trainingX.csv', 'w') as csvfile:  # 训练数据
        writer = csv.writer(csvfile)
        samples = np.array(trX2 * 8, dtype=float)
        writer.writerows(samples.reshape([-1, 576]))

    with open('predict dataset/trainingY.csv', 'w') as csvfile:  # 训练分类标签
        writer = csv.writer(csvfile)
        samples = np.array(trY2, dtype=float)
        writer.writerows(samples)

    with open('predict dataset/testingX.csv', 'w') as csvfile:  # 测试数据
        writer = csv.writer(csvfile)
        samples = np.array(teX * 8, dtype=float)
        writer.writerows(samples.reshape([-1, 576]))

    with open('predict dataset/testingY.csv', 'w') as csvfile:  # 测试标签
        writer = csv.writer(csvfile)
        samples = np.array(teY, dtype=float)
        writer.writerows(samples)

    with open('predict dataset/oneday_ahead_datasets.csv', 'r') as csvfile:  # 日前预测数据
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    forecastX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    m = np.clip(m, 0, 8.0)
    print("###数据集中的峰值", m)
    print(shape(rows))
    for x in range(52):
        train = rows[:-288, x].reshape(-1, 576)
        train = train / 8

        # print(shape(train))
        if forecastX == []:
            forecastX = train
        else:
            forecastX = np.concatenate((forecastX, train), axis=0)
    print("Shape ForecastX----", shape(forecastX))
    forecastX = forecastX[index[8464:]]
    forecastX = forecastX.reshape([-1, 576])
    return trX2, trY2, teX, teY, forecastX


"""
模型准备：
使用W-DCGAN模型和一些辅助函数
使用优化模型来完成场景预测的gradient
"""


def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:  # 四维数组的归一化
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b

    elif X.get_shape().ndims == 2:  # 二维数组的归一化
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X - mean), 0)
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X * g + b

    else:
        raise NotImplementedError

    return X


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=o, logits=t))


class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[24, 24, 1],
            dim_z=100,
            dim_y=12,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            lam=0.05
    ):
        self.lam = lam
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal([dim_z + dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1 + dim_y, dim_W2 * 6 * 6], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2 + dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5, 5, dim_channel, dim_W3 + dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal([5, 5, dim_channel + dim_y, dim_W3], stddev=0.02),
                                      name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3 + dim_y, dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2 * 6 * 6 + dim_y, dim_W1], stddev=0.02),
                                      name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1 + dim_y, 1], stddev=0.02), name='discrim_W4')

    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        pred_high = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        pred_low = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        h4 = self.generate(Z, Y)
        # image_gen来自生成器的sigmoid输出
        image_gen = tf.nn.sigmoid(h4)

        raw_real2 = self.discriminate(image_real, Y)
        # p_real = tf.nn.sigmoid(raw_real)
        p_real = tf.reduce_mean(raw_real2)

        raw_gen2 = self.discriminate(image_gen, Y)
        # p_gen = tf.nn.sigmoid(raw_gen)
        p_gen = tf.reduce_mean(raw_gen2)

        discrim_cost = tf.reduce_mean(raw_real2) - tf.reduce_mean(raw_gen2)  # W距离
        gen_cost = -tf.reduce_mean(raw_gen2)

        mask = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='mask')
        # ^Ppred[32,24,24,1]

        # 场景预测模型中的约束的变体log barriers
        contextual_loss_latter = tf.contrib.layers.flatten(
            -tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.multiply(
                    tf.ones_like(mask) - mask, image_gen))  # image_gen也就是G(z)
            - tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.multiply(
                    tf.ones_like(mask) - mask, pred_low)))  # pred_high和pred_low分别表示预测区间的上下限
        # 点预测部分的优化目标函数
        contextual_loss_latter = tf.where(tf.is_nan(contextual_loss_latter), tf.ones_like(
            contextual_loss_latter) * 1000000.0, contextual_loss_latter)

        contextual_loss_latter2 = tf.reduce_sum(contextual_loss_latter, 1)
        # 历史数据部分的优化目标函数
        contextual_loss_former = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(mask, image_gen) - tf.multiply(mask, image_real))), 1)
        # 不带约束log目标模型的gradients
        contextual_loss_prepare = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(
                tf.ones_like(mask) - mask, image_real))), 1)  # 这里的mask扮演的作用是什么呢?
        perceptual_loss = gen_cost
        # 完整的优化任务的目标函数，其中β和γ都为0.05
        complete_loss = contextual_loss_former + self.lam * perceptual_loss + 0.05 * contextual_loss_latter2
        # 带有约束log目标模型的gradients
        grad_complete_loss = tf.gradients(complete_loss, Z)
        # 不带约束log目标模型的gradients
        grad_uniform_loss = tf.gradients(contextual_loss_prepare, Z)
        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen, grad_complete_loss, \
               pred_high, pred_low, mask, contextual_loss_latter, contextual_loss_former, grad_uniform_loss

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])], 3)

        h1 = lrelu(tf.nn.conv2d(X, self.discrim_W1, strides=[1, 2, 2, 1], padding='SAME'))
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)

        h2 = lrelu(batchnormalize(tf.nn.conv2d(h1, self.discrim_W2, strides=[1, 2, 2, 1], padding='SAME')))
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        discri = tf.matmul(h2, self.discrim_W3)
        h3 = lrelu(batchnormalize(discri))
        return h3

    def generate(self, Z, Y):
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat([Z, Y], 1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size, 6, 6, self.dim_W2])
        h2 = tf.concat([h2, yb * tf.ones([self.batch_size, 6, 6, self.dim_y])], 3)

        output_shape_l3 = [self.batch_size, 12, 12, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(batchnormalize(h3))
        h3 = tf.concat([h3, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [self.batch_size, 24, 24, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        return h4

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat([Z, Y], 1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size, 6, 6, self.dim_W2])
        h2 = tf.concat([h2, yb * tf.ones([batch_size, 6, 6, self.dim_y])], 3)

        output_shape_l3 = [batch_size, 12, 12, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(batchnormalize(h3))
        h3 = tf.concat([h3, yb * tf.ones([batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [batch_size, 24, 24, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        x = tf.nn.sigmoid(h4)
        return Z, Y, x


"""
模型训练：
场景生成部分依旧使用W-DGGAN模型
场景预测部分则是由找到z合适的初始点的迭代以及场景预测生成的迭代组成
"""

n_epochs = 50
learning_rate = 0.0002
batch_size = 32
image_shape = [24, 24, 1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1
k = 3

trX, trY, teX, teY, forecastX = load_solar_data_new()  #
print("###训练样本的特征形状: ", shape(trX))
print("-----光伏数据载入完毕-----")


def construct(X):  # 用于构建约束区间
    X_new1 = np.copy(X[:, 288:576])  # 提取后半段数据
    X_new_high = [x * 2.5 for x in X_new1]  # X*1.2     α=1.2
    X_new_low = [x * 0.4 for x in X_new1]  # X*0.8
    x_samples_high = np.concatenate((X[:, 0:288], X_new_high), axis=1)  # 拼成批量完整的场景
    x_samples_high = np.clip(x_samples_high, 0.05, 0.95)  # 限制范围
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)  # 拼成批完整的场景
    x_samples_low = np.clip(x_samples_low, 0.05, 0.9)  # 限制范围
    return x_samples_high, x_samples_low


def construct2(X):
    X_new = X[:, 288:576]  # 提取后半段数据
    X_new_high = [x * 2 for x in X_new]  # X*2.5  α=2.5
    X_new_low = [x * 0.5 for x in X_new]  # X*0.4
    X_new_high = np.clip(X_new_high, 0.16, 1)  # 限制范围
    x_samples_high = np.concatenate((X[:, 0:288], X_new_high), axis=1)  # 拼成批完整的场景
    X_new_low = np.clip(X_new_low, 0, 0.6)  # 限制范围
    x_samples_low = np.concatenate((X[:, 0:288], X_new_low), axis=1)  # 拼成批完整的场景
    return x_samples_high, x_samples_low


dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    # W1,W2,W3: 卷积层的维数
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    dim_W3=dim_W3,
)
print("------DGAN模型载入完毕------")

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, \
complete_loss, high_tf, low_tf, mask_tf, log_loss, loss_former, loss_prepare = dcgan_model.build_model()

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-d_cost_tf, var_list=discrim_vars))
train_op_gen = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(g_cost_tf, var_list=gen_vars))
Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=batch_size)

Z_np_sample = np.random.uniform(-1, 1, size=(batch_size, dim_z))
Y_np_sample = OneHot(np.random.randint(5, size=[batch_size]), n=12)
iterations = 0
P_real = []
P_fake = []
P_distri = []
discrim_loss = []

with tf.Session() as sess:
    # 进入会话，开始进行训练
    init = tf.global_variables_initializer()  # 将会话中所有参数初始化
    sess.run(init)
    saver = tf.train.Saver()

    print("###每个epoch中的批次数:", len(trY) / batch_size)
    for epoch in range(n_epochs):
        print("epoch" + str(epoch))

        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]
        trY2 = OneHot(trY, n=12)  # 对标签进行编码
        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
        ):

            Xs = trX[start:end].reshape([-1, 24, 24, 1])
            Ys = trY2[start:end]  # [start:start+32]
            # 使用均匀分布数据生成噪声样本
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

            # 对于每一次的迭代，分别生成D和G，k=3
            if np.mod(iterations, k) != 0:
                _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys
                    })


            else:
                _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        Z_tf: Zs,
                        Y_tf: Ys,
                        image_tf: Xs
                    })
            p_real_val, p_gen_val = sess.run([p_real, p_gen], feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            P_real.append(p_real_val.mean())
            P_fake.append(p_gen_val.mean())

            if np.mod(iterations, 2000) == 0:
                print("iterations---- ", iterations)
                gen_loss_val, discrim_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, d_cost_tf, p_real, p_gen],
                                                                                 feed_dict={Z_tf: Zs, image_tf: Xs,
                                                                                            Y_tf: Ys})
                print("###Average P(real)=", p_real_val.mean())
                print("###Average P(gen)=", p_gen_val.mean())
                print("###discrim loss:", discrim_loss_val)
                print("###gen loss:", gen_loss_val)

                Z_np_sample = np.random.uniform(-1, 1, size=(batch_size, dim_z))
                generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample: Z_np_sample,
                        Y_tf_sample: Y_np_sample
                    })
                generated_samples = generated_samples.reshape([-1, 576])
                generated_samples = generated_samples * 8

                with open('2.%s.csv' % iterations, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(generated_samples)
            print(iterations)
            iterations = iterations + 1

    print("------开始场景优化------")
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.001  # 学习率0.001
    iterations = 0

    completed_samples = []
    # mask顾名思义：面具，掩盖；一个batchsize*24*24*1的矩阵，作者在训练时进行Tensor实例化赋给mask的值
    # 是一个前12行全1，后12行全0的矩阵，mask和风电场景矩阵(batchsize*24*24*1)相乘的结果是前12行保留原有
    # 值大小，后12行变为全0（把后12行“盖住了”），即提取了前一天光伏电数据，同理，mask的互补矩阵tf.ones_like(mask) – mask
    # 也是一个batchsize*24*24*1的矩阵，只不过前12行全0，后12行全1，用于提取后一天的光伏电数据。mask和预测数据矩阵相乘同理。
    # 这个mask是控制预测范围的变量，比如赋给mask的值是一个前23行全1，后1行全0的矩阵，那么此时的场景预测范围是最后2小时。
    mask = np.ones([batch_size, 24, 24, 1])  # shape[32,24,24,1]的全1数组，用来
    mask[:, 12:24, :, :] = 0.0  #

    for start, end in zip(
            range(0, len(forecastX), batch_size),
            range(batch_size, len(forecastX), batch_size)
    ):
        print("准备在迭代过程中生成预测场景----%s", iterations)
        forecast_samples = forecastX[start:end]  # 从日前预测数据中提取一批
        Xs = teX[start:end]  # 对训练好的模型进行测试
        X_feed_high, X_feed_low = construct(forecast_samples)
        X_feed_high2, X_feed_low2 = construct2(forecast_samples)
        Ys = teY[start:end]  # 测试数据标签
        Ys = OneHot(Ys, n=12)  # 标签编码，方便机器操作

        with open('predict dataset/orig_iter%s.csv' % iterations, 'w') as csvfile:  # 测试数据集
            writer = csv.writer(csvfile)
            orig_samples = Xs * 8
            writer.writerows(orig_samples)
        with open('predict dataset/%s.csv' % iterations, 'w') as csvfile:  # 日前预测集中提取的数据
            writer = csv.writer(csvfile)
            orig_samples = forecast_samples * 8
            writer.writerows(orig_samples)
        with open('predict dataset/forhigh_iter%s.csv' % iterations, 'w') as csvfile:  # 预测区间上限
            writer = csv.writer(csvfile)
            orig_samples = X_feed_high2 * 8
            writer.writerows(orig_samples)
        with open('predict dataset/forlow_iter%s.csv' % iterations, 'w') as csvfile:  # 预测区间下限
            writer = csv.writer(csvfile)
            orig_samples = X_feed_low2 * 8
            writer.writerows(orig_samples)

        Xs_shaped = Xs.reshape([-1, 24, 24, 1])
        samples = []

        for batch in range(40):
            print("批次:", batch)
            zhats = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)  # z为(-1,1)范围内随机抽样
            image_pre = np.zeros([batch_size, 576])  # [32,576]全零数组
            # 对区间内进行随机采样，得到一个Pinitial
            for i in range(batch_size):
                for j in range(288, 576):
                    image_pre[i][j] = np.random.uniform(X_feed_low[i, j], X_feed_high[i, j])

            image_pre = image_pre.reshape([-1, 24, 24, 1])
            m = 0
            v = 0
            for i in range(1200):  # 初始测点迭代
                fd = {
                    Z_tf: zhats,
                    image_tf: image_pre,  # P initial
                    Y_tf: Ys,
                    mask_tf: mask,
                }
                # MomentumGD(z,gz)
                g, = sess.run([loss_prepare], feed_dict=fd)  # 这里不加逗号会出现矩阵相乘时类型不匹配的问题，参数更新
                # 对Z_tf进行调整更新
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)  # 将zhats限制在[-1，1]之间

            image_pre = image_pre.reshape([-1, 576])

            m = 0
            v = 0

            for i in range(1000):  # 优化问题迭代
                fd = {
                    Z_tf: zhats,
                    image_tf: Xs_shaped,
                    Y_tf: Ys,
                    high_tf: X_feed_high2.reshape([-1, 24, 24, 1]),
                    low_tf: X_feed_low2.reshape([-1, 24, 24, 1]),  # 完整的loss需要区间上下限
                    mask_tf: mask,
                }
                # log_loss_value=log部分的loss，loss_former=内积部分的loss
                g, log_loss_value, sample_loss_value = sess.run([complete_loss, log_loss, loss_former], feed_dict=fd)
                # 对Z_tf进行调整更新
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = beta1 * m_prev + (1 - beta1) * g[0]
                v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))

                zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                zhats = np.clip(zhats, -1, 1)

                if np.mod(i, 200) == 0:
                    print("###梯度迭代次数:", i)
                    print("###对数损失指标：", log_loss_value[0])
                    print("###样本损失指标:", sample_loss_value)

            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: zhats,
                    Y_tf_sample: Ys
                })  # 生成预测场景

            generated_samples = generated_samples.reshape(32, 576)  # 塑形
            samples.append(generated_samples)

        samples = np.array(samples, dtype=float)

        print(shape(samples))
        samples = samples * 8
        plt.plot(generated_samples[0]*8,color='r')
        plt.plot(X_feed_low[0]*8, color='g')
        plt.plot(X_feed_high[0]*8, color='y')
        plt.plot(orig_samples[0], color='b')
        plt.show()
        with open('generated_iteration-%s.csv' % iterations, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(samples.reshape([-1, 576]))  # 得到了理想的预测场景
        iterations += 1
