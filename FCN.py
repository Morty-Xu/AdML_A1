import random
import numpy as np
from mnist_loader import mnist_dataloader
from util import currentTime
import joblib
import time
from visualdl import LogWriter
from Cost import QuadraticCost, CrossEntropyCost


def sigmoid(z):
    """
    sigmoid函数
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    sigmoid函数的导数
    """
    return sigmoid(z) * (1 - sigmoid(z))


class FCN(object):
    """
    全连接神经网络
    """
    def __init__(self, sizes, cost):
        """
        :param sizes: 是一个列表，其中包含了神经网络每一层的神经元的个数，列表的长度就是神经网络的层数。
                神经网络的权重和偏置是随机生成的，使用一个均值为0，方差为1的高斯分布。
        :param cost: 网络所选用的代价函数
        """
        self.cost = cost
        self.sizes = sizes
        self._num_layers = len(sizes)
        self.biases = None
        self.weights = None
        self.default_weight_initializer()

    def default_weight_initializer(self, large=False):
        """
        初始化权重和偏置
        :param large:
        :return:
        """
        # 为隐藏层和输出层生成偏置向量b，以[784,30,10]为例，那么一共会生成2个偏置向量b，分别属于隐藏层和输出层，大小分别为30x1,10x1。
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 为隐藏层和输出层生成权重向量W, 以[784,30,10]为例，这里会生成2个权重向量w，分别属于隐藏层和输出层，大小分别是30x784, 10x30。
        if not large:
            self.weights = [np.random.randn(y, x) / np.sqrt(x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        else:
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        前向计算，返回神经网络的输出。公式如下:
        a = sigmoid(w*x+b)
        :param a: 神经网络的输入
        :return: 神经网络的输出
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, validation_data, epochs, mini_batch_size, lr,
            penalty=None,
            lmbda=0.0,
            monitor_validation_cost=False,
            monitor_validation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            generate_VisualDL=False):
        """
        使用小批量样本的随机梯度下降法(mini-batch GD) 更新参数和偏置
        :param training_data: 训练集
        :param validation_data: 验证集数据
        :param epochs: 训练世代
        :param mini_batch_size: batch_size
        :param lr: 学习率
        :param penalty: 正则项， None：无正则项；L1：L1正则；L2：L2正则
        :param lmbda: 正则化参数
        :param monitor_validation_cost:
        :param monitor_validation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :return:
        """

        n_val = len(validation_data)
        n = len(training_data)
        validation_cost, validation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, lr, penalty, lmbda, len(training_data)
                )

            print("Epoch %s training complete: " % (j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, penalty, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {0}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("accuracy rate on training data: %.2f%%" % (accuracy / n * 100))
            if monitor_validation_cost:
                cost = self.total_cost(validation_data, penalty, lmbda)
                validation_cost.append(cost)
                print("Cost on validation data: {0}".format(cost))
            if monitor_validation_accuracy:
                accuracy = self.accuracy(validation_data)
                validation_accuracy.append(accuracy)
                print("accuracy rate on validation data: %.2f%%" % (accuracy / n_val * 100))

        if generate_VisualDL:
            # 使用可视化工具VisualDL跟踪训练过程，https://github.com/PaddlePaddle/VisualDL
            with LogWriter(logdir="./log_FCN/train") as writer:
                for step in range(epochs):
                    writer.add_scalar(tag="train/acc", step=step, value=float('%.4f' % (training_accuracy[step] / n)))
                    writer.add_scalar(tag="train/cost", step=step, value=training_cost[step])

            with LogWriter(logdir="./log_FCN/val") as writer:
                for step in range(epochs):
                    writer.add_scalar(tag="train/acc", step=step, value=float('%.4f' % (validation_accuracy[step] / n_val)))
                    writer.add_scalar(tag="train/cost", step=step, value=validation_cost[step])
        return validation_cost, validation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, lr, penalty, lmbda, n):
        """
        使用mini_batch GD方式的条件下，更新一个batch的参数和偏置
        :param mini_batch: 一个tuple (x,y) 的列表
        :param lr: 学习率
        :param penalty: 正则项， None：无正则项；L1：L1正则；L2：L2正则
        :param lmbda: 正则化参数
        :param n: 训练集大小
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 反向传播算法，运用链式法则求得对b和w的偏导
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 对小批量训练数据集中的每一个求得的偏导数进行累加
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降得出的规则来更新权重和偏置向量
        if penalty == 'L1':
            self.weights = [(1 - lr * (lmbda / n)) * np.sign(w) - (lr / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
        elif penalty == 'L2':
            self.weights = [(1 - lr * (lmbda / n)) * w - (lr / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
        else:
            self.weights = [w - (lr / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        return 0

    def backprop(self, x, y):
        """
        反向传播算法，计算损失对w和b的梯度
        :param x: 训练数据x
        :param y: 训练数据x对应的标签
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，计算网络的输出
        activation = x
        # 一层一层存储全部激活值的列表
        activations = [x]
        # 一层一层地存储全部的z向量，即带权输入
        zs = []
        for b, w in zip(self.biases, self.weights):
            # 利用 z = wT*x+b 依次计算网络的输出
            z = np.dot(w, activation) + b
            zs.append(z)
            # 将每个神经元的输出z通过激活函数sigmoid
            activation = sigmoid(z)
            # 将激活值放入列表中暂存
            activations.append(activation)

        # 反向传播过程
        # 首先计算输出层的误差delta L
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self._num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def total_cost(self, data, penalty, lmbda):
        """
        计算整体损失值
        :param data: （x, y)的tuple形式，x为输入数据，y为对应标签
        :param lmbda: 正则化参数
        :param convert: 如data是训练集，则置为False；如data是验证集或测试集 ，则置为True
        :return: 返回损失值
        """

        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)

            cost += self.cost.fn(a, y) / len(data)
        if penalty == 'L1':
            for t in [np.abs(w) for w in self.weights]:
                cost += (lmbda / len(data)) * np.sum(t)
        elif penalty == 'L2':
            cost += 0.5 * (lmbda / len(data)) * sum(
                np.linalg.norm(w) ** 2 for w in self.weights)

        return cost

    def accuracy(self, data):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """

        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                   for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def save(self, save_root):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}

        joblib.dump(data, save_root + "weights_" + currentTime() + ".pt")


if __name__ == "__main__":
    # minst数据集路径
    fn = './MNIST_Dataset'

    train_data, val_data = mnist_dataloader(fn)
    startime = time.perf_counter()
    try:
        fc = FCN([784, 100, 10], cost=CrossEntropyCost)
        # 设置迭代次数20次，mini-batch大小为10，学习率为0.5，并且设置测试集，即每一轮训练完成之后，都对模型进行一次评估。
        fc.SGD(train_data, val_data, 20, 10, 0.5, 'L2', 0.1,
               monitor_validation_accuracy=True,
               monitor_validation_cost=True,
               monitor_training_accuracy=True,
               monitor_training_cost=True,
               generate_VisualDL=True)
        fc.save("./")

    except Exception as e:
        endtime = time.perf_counter()
        print("Using {} s".format(endtime - startime, '.2f'))
        raise e




