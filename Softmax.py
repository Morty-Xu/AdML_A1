import numpy as np
import time
from mnist_loader import mnist_dataloader
from Cost import CrossEntropyCost
from util import currentTime
import random
import joblib
from visualdl import LogWriter


class Softmax(object):
    def __init__(self, sizes):
        """
        :param size: [x, y] x为输入维度， y为输出维度
        """
        self.sizes = sizes
        self.biases = None
        self.biases_shape = None
        self.weights = None
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        初始化权重和偏置
        :return:
        """
        # 生成偏置 b
        self.biases = np.random.randn(self.sizes[1], 1)
        self.biases_shape = self.biases.shape
        # 生成权重 w
        self.weights = np.random.randn(self.sizes[0], self.sizes[1])

    def train(self, training_data, validation_data, num_epoch, mini_batch_size, lr, penalty, lmbda,
                monitor_validation_cost = False,
                monitor_validation_accuracy = False,
                monitor_training_cost = False,
                monitor_training_accuracy = False,
                generate_VisualDL = False):
        """
        训练数据
        :param training_data: 训练集
        :param validation_data: 验证集
        :param num_epoch: epoch数
        :param mini_batch_size: batch_size
        :param lr:  学习率
        :param penalty: 正则化。'L1';'L2';'None'
        :param lmbda:
        :param monitor_validation_cost:
        :param monitor_validation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :param generate_VisualDL:
        :return:
        """

        n_val = len(validation_data)
        n = len(training_data)
        validation_cost, validation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(num_epoch):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
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
            with LogWriter(logdir="./log_Softmax/train") as writer:
                for step in range(num_epoch):
                    writer.add_scalar(tag="train/acc", step=step, value=float('%.4f' % (training_accuracy[step] / n)))
                    writer.add_scalar(tag="train/cost", step=step, value=training_cost[step])

            with LogWriter(logdir="./log_Softmax/val") as writer:
                for step in range(num_epoch):
                    writer.add_scalar(tag="train/acc", step=step, value=float('%.4f' % (validation_accuracy[step] / n_val)))
                    writer.add_scalar(tag="train/cost", step=step, value=validation_cost[step])
        return validation_cost, validation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, lr, penalty, lmbda, n):
        """
        使用mini_batch GD方式的条件下，更新一个batch的参数和偏置
        :param mini_batch: 一个tuple (x,y) 的列表
        :param lr: 学习率
        :param penalty:  正则项， None：无正则项；L1：L1正则；L2：L2正则
        :param lmbda: 正则化参数
        :param n: 训练集大小
        :return:
        """
        nabla_b = np.zeros(self.biases_shape)
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            # 对小批量训练数据集中的每一个求得的偏导数进行累加
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, self.forward(x) - y)]

            # zip(nabla_w, np.dot(x, self.forward(x) - y).transpose())
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, np.dot(x, (self.forward(x) - y).transpose()))]

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

    def softmax(self, x):
        x_exp = np.exp(x)
        partition = x_exp.sum(0, keepdims=True)
        return x_exp / partition  # 这里应用了广播机制

    def forward(self, x):
        """
        前向计算
        :return:
        """
        return self.softmax(np.dot(x.reshape((-1, np.array(self.weights).shape[0])),
                                   self.weights).transpose() + self.biases)

    def accuracy(self, data):
        """
        计算预测正确的数量
        :param data: 多个(x, y)，x为输入数据，y为label
        :return:
        """

        results = [(np.argmax(self.forward(x)), np.argmax(y))
                   for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

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
            a = self.forward(x)

            cost += CrossEntropyCost.fn(a, y) / len(data)
        if penalty == 'L1':
            for t in [np.abs(w) for w in self.weights]:
                cost += (lmbda / len(data)) * np.sum(t)
        elif penalty == 'L2':
            cost += 0.5 * (lmbda / len(data)) * sum(
                np.linalg.norm(w) ** 2 for w in self.weights)

        return cost

    def save(self, save_root):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(CrossEntropyCost.__name__)}

        joblib.dump(data, save_root + "weights_" + currentTime() + ".pt")


if __name__ == "__main__":
    # minst数据集路径
    fn = './MNIST_Dataset'

    train_data, val_data = mnist_dataloader(fn)
    startime = time.perf_counter()
    try:
        model = Softmax(sizes=[784, 10])
        # 设置迭代次数20次，mini-batch大小为10，学习率为0.5，并且设置测试集，即每一轮训练完成之后，都对模型进行一次评估。
        model.train(train_data, val_data, 20, 10, 0.5, 'L1', 0.1,
               monitor_validation_accuracy=True,
               monitor_validation_cost=True,
               monitor_training_accuracy=True,
               monitor_training_cost=True,
               generate_VisualDL=True)

        model.save("./")

    except Exception as e:
        endtime = time.perf_counter()
        print("Using {} s".format(endtime - startime, '.2f'))
        raise e
