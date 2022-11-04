import numpy as np
import joblib
import os
import cv2
from sklearn.metrics import classification_report


def sigmoid(z):
    """
    sigmoid函数
    """
    return 1.0 / (1.0 + np.exp(-z))


def softmax(x):
    x_exp = np.exp(x)
    partition = x_exp.sum(0, keepdims=True)
    return x_exp / partition  # 这里应用了广播机制


def test_dataloader(fn):
    test_data = []
    for image_name in os.listdir(fn):
        name, ext = os.path.splitext(image_name)
        ext = ext[1:]
        if ext == 'jpg' or ext == 'png' or ext == 'bmp':
            lable = name.split('_')[0]
            image = np.array(cv2.imread(os.path.join(fn, image_name),
                                        cv2.IMREAD_GRAYSCALE), dtype='float').reshape(-1, 1) / 255
            test_data.append((image, int(lable)))

    return test_data


def FCN_forward(a, weights, biases):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)

    return a


def Softmax_forward(x, weights, biases):

    return softmax(np.dot(x.reshape((-1, np.array(weights).shape[0])), weights).transpose()
                   + biases)


def report(weights_file, model, testing_data):
    weights = joblib.load(weights_file)['weights']
    biases = joblib.load(weights_file)['biases']
    results = []
    if model == 'FCN':
        results = [[np.argmax(FCN_forward(x, weights, biases)), y]
                   for (x, y) in testing_data]

    elif model == 'Softmax':
        results = [[np.argmax(Softmax_forward(x, weights, biases)), y]
                   for (x, y) in testing_data]

    y_pred = np.array(results)[:, 0]
    y_true = np.array(results)[:, 1]
    return classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])


if __name__ == "__main__":
    fn = './MNIST_Dataset/test_images'
    w_file = './weights_FCN_CE_L2_01.pt'
    test_data = test_dataloader(fn)

    r = report(w_file, 'FCN', test_data)

    print(r)
