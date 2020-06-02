"""
做一个3层的NN来分辨mnist数据集（测试项目）
"""

import numpy
import scipy.special
#import matplotlib.pyplot


# NN 类的定义
class NeuralNetwork:
    # 初始化NN
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 定义隐层、输入输出层神经元数（节点数）
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 链接权重矩阵，whi和who
        # 数组内的权重w_i_j, 其中的ij表示从节点i到节点j的链接
        # w11,w21
        # w12,w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 学习率
        self.lr = learningrate
        # 激活函数是S函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练NN
    def train(self, inputs_list, targets_lists):
        # 将列表转为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_lists, ndmin=2).T
        # 计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 输出层error是预期-实际
        output_errors = targets - final_outputs
        # 隐藏层error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新h-o之间的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # 更新i-h之间的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    # 查询NN
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass

    def actibation_function(self, final_inputs):
        pass


"""
创建一个每层3节点，学习率为0.5的神经网络对象
"""
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.2
# 创建神经网络实例
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 载入训练数据
training_data_file = open("datasets/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练神经网络
epochs = 2
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
# 载入测试数据
test_data_file = open("datasets/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试神经网络
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    # 最大值趋向的标签
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 测试结果
scorecard_array = numpy.asfarray(scorecard)
print("performance", scorecard_array.sum() / scorecard_array.size)
