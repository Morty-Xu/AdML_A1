# AdML_A1：多层感知机

### 问题描述


- Task: handwritten digits recognition (15 points)


- Data: MNIST data set


- In this assignment you will practice putting together a simple image classification pipeline, based on the Softmax and the fully-connected classifier. The goals of this assignment are as follows:

  - understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)

  - understand the train/val/test splits and the use of validation data for hyper-parameter tuning

  - implement and apply a Softmax classifier

  - implement and apply a Fully-connected neural network classifier

  - understand the differences and tradeoffs between these classifiers implement various update rules used to optimize Neural Networks
- Do something extra! 

  - Maybe you can experiment with a different loss function and regularization? (+5 points)
- Or maybe you can experiment with different optimization algorithm (e.g., batch GD, online GD, mini-batch GD, SGD, or other optimization alg., e.g., Momentum, Adsgrad, Adam, Admax)  (+5 points)

### 数据集

jpg格式的MNIST数据集。

下载数据集：链接：https://pan.baidu.com/s/18Fz9Cpj0Lf9BC7As8frZrw 提取码：xhgk

### 使用

本项目实现了两种分类方法：Softmax回归和全连接神经网络

- **Softmax回归**

  - 训练：

  ```
  python Softmax.py
  ```

  - 测试：

  ```
  python test.py #需要调整参数
  ```

- **全连接神经网络**
	- 训练：

  ```
  python FCN.py
  ```

  - 测试：

  ```
  python test.py #需要调整参数
  ```


### 分析

- Softmax + 交叉熵

![SoftmaxCE](/Users/morty/CodeProject/AdML_A1/pictures/SoftmaxCE.png)

- Softmax + 交叉熵 + L1 + 0.1 lmbda

![Softmax_CE_L1_0.1](/Users/morty/CodeProject/AdML_A1/pictures/Softmax_CE_L1_0.1.png)

- Softmax + 交叉熵 + L2 + 0.1 lmbda

![Softmax_CE_L2_01](/Users/morty/CodeProject/AdML_A1/pictures/Softmax_CE_L2_01.png)

- FCN + 均方误差

![FCN_Qu](/Users/morty/CodeProject/AdML_A1/pictures/FCN_Qu.png)

- FCN + 交叉熵

![FCN_CE](/Users/morty/CodeProject/AdML_A1/pictures/FCN_CE.png)

- FCN + 交叉熵 + L1 + 0.1 lmbda

![FCN_CE_L1_01](/Users/morty/CodeProject/AdML_A1/pictures/FCN_CE_L1_01.png)

- FCN + 交叉熵 + L1 + 0.1 lmbda

![FCN_CE_L2_01](/Users/morty/CodeProject/AdML_A1/pictures/FCN_CE_L2_01.png)
