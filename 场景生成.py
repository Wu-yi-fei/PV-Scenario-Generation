"""
GAN的光伏发电场景生成模型

"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 解决个TF库没有编译，不能使用SSE、SSE2、FMA等指令，但是他们是可以加速你的CPU计算的问题

"""
一些辅助函数:
包括一些编码函数和存储函数
"""


def OneHot(X, n, negative_class=0.):  # 定义一个one_hot编码函数
    X = np.asarray(X).flatten()  # 扁平化成一维数组
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class  # 得到一个全零float数组Xoh[len(X),n]
    for i in range(len(X)):  # 这里的编码规则不符合月份的要求，后面将做进一步修改。
        m = X[i]
        Xoh[i, m] = 1
    return Xoh  # 得到归一化的编码矩阵Xoh


# 用于保存类似于图像的场景映射
def save_visualization(X, nh_nw, save_path='out/.jpg'):
    h, w = X.np.shape[1], X.np.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def save(saver, sess, logdir, step):  # 保存模型的save函数
    model_name = 'model'  # 模型名前缀
    checkpoint_path = os.path.join(logdir, model_name)  # 保存路径
    saver.save(sess, checkpoint_path, global_step=step)  # 保存模型
    print('------模型参数已部署------.')


"""
数据准备：
加载光伏历史数据(csv类型文件)到GANs模型，并重新塑成可调的模型输入形状
label只对基于事件的场景生成有用
"""


def my_load_solar():
    # 为基于事件的光伏场景生成进一步创建和处理合适的数据集，
    # 数据来自于NREL的集成数据集，观测地点是加利福尼亚州。
    with open('datasets/solar_datasets.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(np.shape(rows))
    m = np.ndarray.max(rows)
    print("数据集中的最大值为：", m)
    print(np.shape(rows))
    for x in range(rows.shape[1]):
        train = rows[:-288, x].reshape(-1, 576)
        train = train / 8

        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)  # 对应行进行拼接
    print("Shape TrX----", np.shape(trX))

    with open('datasets/solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    label = np.array(rows, dtype=int)
    print("shape Label----", np.shape(label))
    trY = np.tile(label, (52, 1))  # 延展
    print("re-shape Label----", np.shape(trY))
    return trX, trY

def load_solar_data_spatial():
    with open('datasets/spatial.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows = rows.reshape(-1, 576)
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    rows = rows / m
    print("shape spatial----", np.shape(rows))
    return rows
"""
模型准备：
使用WGAN模型和一些辅助函数
"""


def BatchNormalize(X, eps=1e-8, g=None, b=None):  # 归一化操作函数，分辨率1*10^-8,
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)
        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b
    elif X.get_shape().ndims == 2:
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


def LRelu(X, leak=0.2):  # Leak Relu激活函数
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)  # 大于0时x,小于0时leak*x


def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(o, t))


def xavier_init(size):  # 初始化参数时使用的xavier_init函数
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # 初始化标准差
    return tf.random_normal(shape=size, stddev=xavier_stddev)  # 返回初始化的结果


# 一种结合了WGAN和DCGAN的CNN模型

class GAN:  # 创建一个GAN类
    def __init__(
            self,
            batch_size=32,
            image_shape=[24, 24, 1],
            dim_z=100,
            dim_y=6,  # 该参数用于定义事件数量
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,  
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z  # 100
        self.dim_y = dim_y  # 6

        self.dim_W1 = dim_W1  # 1024
        self.dim_W2 = dim_W2  # 128
        self.dim_W3 = dim_W3  # 64
        self.dim_channel = dim_channel
        self.gen_W1 = tf.Variable(tf.random_normal([dim_z + dim_y, dim_W1], stddev=0.02),
                                  name='gen_W1')
        # [100+y，1024]
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1 + dim_y, dim_W2 * 6 * 6],
                                                   stddev=0.02), name='gen_W2')
        # [1024+y,128*6*6]
        self.gen_W3 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2 + dim_y],
                                                   stddev=0.02), name='gen_W3')
        # [5,5,64,128+y]
        self.gen_W4 = tf.Variable(tf.random_normal([5, 5, dim_channel, dim_W3 + dim_y],
                                                   stddev=0.02), name='gen_W4')
        # [5,5,1,64+y]
        self.discrim_W1 = tf.Variable(tf.random_normal([5, 5, dim_channel + dim_y, dim_W3],
                                                       stddev=0.02), name='discrim_W1')
        # [5,5,1+y,64]
        self.discrim_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3 + dim_y, dim_W2],
                                                       stddev=0.02), name='discrim_W2')
        # [5,5,64+y,128]
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2 * 6 * 6 + dim_y, dim_W1],
                                                       stddev=0.02), name='discrim_W3')
        # [128*6*6+y,1024]
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1 + dim_y, 1], stddev=0.02),
                                      name='discrim_W4')
        # [1024+y,1]

    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])  # [32,100]32列100行噪声矩阵
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])  # [32,y]特定事件矩阵
        image_real = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        # 尺寸[32,24,24,1]的时序特征矩阵

        h4 = self.generate(Z, Y)
        # image_gen来自生成器的sigmoid输出
        image_gen = tf.nn.sigmoid(h4)

        raw_real2 = self.discriminate(image_real, Y)
        p_real = tf.reduce_mean(raw_real2)  # 生成数据的reduce mean

        raw_gen2 = self.discriminate(image_gen, Y)
        p_gen = tf.reduce_mean(raw_gen2)  # 真实数据的reduce mean

        discrim_cost = tf.reduce_sum(raw_real2) - tf.reduce_sum(raw_gen2)  # 边际分布的差值为W距离
        gen_cost = -tf.reduce_mean(raw_gen2)  #

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        print("--------开始判别器初始化--------")
        print("Y shape----", Y.get_shape())
        # [32,y]
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        # [32,1,1,y]
        print("image shape----", image.get_shape())
        print("yb shape----", yb.get_shape())
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])], 3)
        # [32,24,24,1+y]
        print("X shape----", X.get_shape())
        h1 = LRelu(tf.nn.conv2d(X, self.discrim_W1, strides=[1, 2, 2, 1], padding='SAME'))
        # [5,5,1,64+y]得到的feature map为[32,12,12,64]
        print("hiden1 shape----", h1.get_shape())
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)
        print("hiden1_concat shape----", h1.get_shape())
        h2 = LRelu(BatchNormalize(tf.nn.conv2d(h1, self.discrim_W2, strides=[1, 2, 2, 1], padding='SAME')))
        # [32,6,6,128]
        print("hiden2 shape----", h2.get_shape())
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        discri = tf.matmul(h2, self.discrim_W3)
        print("discri shape----", discri.get_shape())
        h3 = LRelu(BatchNormalize(discri))
        return h3

    def generate(self, Z, Y):
        print("--------开始生成器初始化--------")
        print("#####噪声Z的特征形状：", Z.get_shape())
        print("#####噪声Y的特征形状：", Y.get_shape())

        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        # [32,1,1,y]
        Z = tf.concat([Z, Y], 1)
        # 组合之后的形状[32,100+y]
        print("合并事件y后的Z的特征形状", Z.get_shape())
        h1 = tf.nn.relu(BatchNormalize(tf.matmul(Z, self.gen_W1)))
        # 噪声Z通过权重W1后进行归一化和激活函数Relu的第一个节点处理后得到h1 [32,1024]
        print("hiden1 shape----", h1.get_shape())
        h1 = tf.concat([h1, Y], 1)
        # 组合之后的形状[32,1024+y]
        print("hiden1_concat shape----", h1.get_shape())
        h2 = tf.nn.relu(BatchNormalize(tf.matmul(h1, self.gen_W2)))
        # 经过第二个隐层的h2 [32,128*6*6]
        print("hiden2 shape----", h2.get_shape())
        h2 = tf.reshape(h2, [self.batch_size, 6, 6, self.dim_W2])
        # [32,6,6,128]
        print("hiden2_reshape shape----", h2.get_shape())
        h2 = tf.concat([h2, yb * tf.ones([self.batch_size, 6, 6, self.dim_y])], 3)
        # [32,6,6,128+y]
        n = yb * tf.ones([self.batch_size, 6, 6, self.dim_y])
        # [32,6,6,y]
        print("shape of yb new----", n.get_shape())
        print("hiden2_concat shape----", h2.get_shape())

        output_shape_l3 = [self.batch_size, 12, 12, self.dim_W3]
        # 实现用32批次的134通道6*6图像，用一个5×5的卷积核（对应的shape：[5，5，64，128+6]）去做卷积，得到一
        # 个12*12的feature map，做一个反卷积。
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3,
                                    strides=[1, 2, 2, 1])
        # [32,12,12,64]
        h3 = tf.nn.relu(BatchNormalize(h3))
        print("hiden3 shape----", h3.get_shape())
        h3 = tf.concat([h3, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)
        # [32,12,12,64+y]
        print("hiden3_concat shape----", h3.get_shape())

        output_shape_l4 = [self.batch_size, 24, 24, self.dim_channel]
        # 实现用32批次的134通道12*12图像，，用一个5×5的卷积核（对应的shape：[5，5，64，128+y]）去做卷积，得到
        # 一个12*12的feature map，做一个反卷积。
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4,
                                    strides=[1, 2, 2, 1])
        # [32,24,24,1]
        return h4

    def samples_generator(self, batch_size):  # 这个生成器是使用现阶段的训练权重参数来生成新的样本集，
        # 输出csv文件
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat([Z, Y], 1)
        h1 = tf.nn.relu(BatchNormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(BatchNormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size, 6, 6, self.dim_W2])
        h2 = tf.concat([h2, yb * tf.ones([batch_size, 6, 6, self.dim_y])], 3)

        output_shape_l3 = [batch_size, 12, 12, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(BatchNormalize(h3))
        h3 = tf.concat([h3, yb * tf.ones([batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [batch_size, 24, 24, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        x = tf.nn.sigmoid(h4)
        return Z, Y, x


"""
模型训练：
使用W-DGGAN模型
"""

n_epochs = 80  # 训练遍历次数
learning_rate = 0.0002  # 学习率
batch_size = 32  # batch数据载量
image_shape = [24, 24, 1]  # 喂入model数据的尺寸
dim_z = 100  # 噪声尺寸
dim_W1 = 1024  # 输入层
dim_W2 = 128  # 中间层1
dim_W3 = 64  # 中间层2
dim_channel = 1  # 通道数
mu, sigma = 0, 0.1  # 高斯分布
events_num = 12  # 事件数量

visualize_dim = 32
generated_dim = 32

# 载入训练数据以及标签
trX, trY = my_load_solar()
# trX, trY=load_solar_data()
# trX, trY=load_solar_data_spatial()

print("训练样本的特征形状为： ", np.shape(trX))
print("---------训练数据载入完毕---------")

dcgan_model = GAN(
    dim_y=events_num  # 根据事件的数量更改参数
    # 在这里可以修改模型参数
    # dim_z:输入噪声的维数
    # W1,W2,W3:卷积层的维数
)
print("----------W_DCGAN模型初始化完毕----------")

# Z_tf,Y_tf: 占位符
# image_tf: 图像占位符
# d_cost_tf, g_cost_tf: 鉴别器和发电机的成本
# p_real, p_gen: 鉴别器的输出来判断真实的/生成的

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()  # 创建一个交互式Session
saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)  # 保存最近的10个模型
if not os.path.exists('out/'):  # 初始化训练过程中的可视化结果的输出文件夹
    os.makedirs('out/')

if not os.path.exists('snapshots/'):  # 初始化训练过程中的模型保存文件夹
    os.makedirs('snapshots/')

# 初始化模型参数
discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]

train_op_discrim = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-d_cost_tf,
                                                                           var_list=discrim_vars))  # 判别器的训练器
train_op_gen = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(g_cost_tf,
                                                                       var_list=gen_vars))  # 生成器的训练器
Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(
    batch_size=visualize_dim)
tf.global_variables_initializer().run()

Zs = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)
Y_np_sample = OneHot(np.random.randint(events_num, size=[visualize_dim]), n=events_num)
iterations = 0
k = 3  # 控制D和G的相对训练次数，保持训练过程平衡

gen_loss_all = []
P_real = []
P_fake = []
P_distri = []
discrim_loss = []

# 开始训练

for epoch in range(n_epochs):
    print("epoch世代----" + str(epoch))
    index = np.arange(len(trY))
    np.random.shuffle(index)
    trX = trX[index]
    trY = trY[index]
    trY2 = OneHot(trY, n=events_num)

    for start, end in zip(
            range(0, len(trY), batch_size),
            range(batch_size, len(trY), batch_size)
    ):

        Xs = trX[start:end].reshape([-1, 24, 24, 1])
        Ys = trY2[start:end]

        # 使用均匀分布或高斯分布的噪声数据作为GANs模型输入
        Zs = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)  # [32,100]

        # 对于每次迭代，都分别生成D和G, 交替次数k=3
        if np.mod(iterations, k) == 0:
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys,
                    image_tf: Xs
                })
            discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf, p_real, p_gen],
                                                               feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            print("####迭代次数:", iterations)

        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys,
                    image_tf: Xs
                })

            gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen],
                                                           feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
        P_real.append(p_real_val.mean())
        P_fake.append(p_gen_val.mean())
        discrim_loss.append(discrim_loss_val)

        if np.mod(iterations, 100) == 0:
            save(saver, sess, 'snapshots/', iterations)
            print("迭代次数= ", iterations)
            print("#####Average P(real)=", p_real_val.mean())
            print("#####Average P(gen)=", p_gen_val.mean())
            print("#####Discrim loss:", discrim_loss_val)
        if np.mod(iterations, 1000) == 0:
            Y_np_sample = OneHot(np.random.randint(12, size=[visualize_dim]), n=events_num)
            Z_np_sample = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)
            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: Z_np_sample,
                    Y_tf_sample: Y_np_sample
                })
            generated_samples = generated_samples.reshape([-1, 576])
            generated_samples = generated_samples * 8
            with open('%s.csv' % iterations, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(generated_samples)

        iterations += 1

Y_np_sample = OneHot(np.random.randint(12, size=[visualize_dim]), n=events_num)
Zs = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)
generated_samples = sess.run(
    image_tf_sample,
    feed_dict={
        Z_tf_sample: Z_np_sample,
        Y_tf_sample: Y_np_sample
    })
generated_samples = generated_samples.reshape([-1, 576])
generated_samples = generated_samples * 8
with open('sample.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(generated_samples)
with open('label.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(Y_np_sample)

# 图表展示真实场景与生成场景的loss
print("P_real：", P_real)
print("P_fake：", P_fake)

plt.plot(P_real, label="real distribution")
plt.plot(P_fake, label="fake discribution")
plt.legend()
plt.show()

plt.plot(discrim_loss, label="Discrim_loss")
plt.legend()
plt.show()
