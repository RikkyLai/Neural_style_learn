import scipy.io
import cv2
import numpy as np
import tensorflow as tf


vgg_path = 'imagenet-vgg-verydeep-19.mat'
layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
data = scipy.io.loadmat(vgg_path)
mean = data["normalization"][0][0][0][0][0]
mean = [mean[1], mean[2], mean[0]]


# 读取图片，resize
def imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.array([preprocess(image, mean)]).astype(np.float32)
    return image


# 归一化操作
def preprocess(image, mean_pixel):
    return image-mean_pixel


# 还原 归一化操作
def depreprocess(image):
    return image+mean


# 根据网络结构  输出每一层经过隐藏层的结果，结果为特征图
def vgg19_network(input_image):

    net = input_image
    network = {}
    for i, name in enumerate(layers):
        if 'conv' in name:
            weights, bias = data['layers'][0][i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))    # 核 的 参数矩阵和我们定义的矩阵长宽反转
            net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, bias.reshape(-1))

        elif 'relu' in name:
            net = tf.nn.relu(net)

        elif 'pool' in name:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        network[name] = net

    return network




