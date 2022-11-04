import numpy as np


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


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        # 均方差误差函数
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        # 均方差误差函数的导数，其中激活函数为sigmoid
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log_FCN_QCost(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        交叉熵损失函数定义 C = -1/n ∑(y*lna + (1-y)ln(1-a))
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return a-y