from network import imread
from network import vgg19_network
from network import depreprocess
import numpy as np
import tensorflow as tf
import cv2


content_path = 'examples/aaa.jpg'
style_path = 'examples/the_scream.jpg'

content_layers = ['relu5_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


# 训练
def stylize(content_image_path, style_image_path, learning_rate=5, epoch=1000):
    content_image = imread(content_image_path)  # 加载内容图片
    style_image = imread(style_image_path)      # 加载风格图片
    target = tf.Variable(tf.random_normal(content_image.shape), dtype=tf.float32)  # 一直要训练的目标图
    style_input = tf.constant(style_image, dtype=tf.float32)     # 转化为tf格式  网络输入
    content_input = tf.constant(content_image, dtype=tf.float32)

    cost = total_loss(target, style_input, content_input)     #  计算cost

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            _, cost_value, target_image = sess.run([train_op, cost, target])

            if i % 100 == 0:
                cv2.destroyAllWindows()
                target_image = depreprocess(target_image)       # 将归一化的图片还原
                image = np.clip(target_image, 0, 255).astype(np.uint8)
                image = np.squeeze(image)
                print('%d epoch' % i)
                print('cost value: %f' % cost_value)
                cv2.imshow('result', image)
                cv2.waitKey(100)


def total_loss(target, style_input, content_input):
    target_features = vgg19_network(target)       # 得到 目标图片经过网络后的 特征图
    style_features = vgg19_network(style_input)   # 得到 风格图片经过网络后的 特征图
    content_features = vgg19_network(content_input)   # 得到 内容图片经过网络后的 特征图
    content_loss = 0
    style_loss = 0
    for i, name in enumerate(content_layers):
        content_loss += compute_content_loss(content_features[name], target_features[name])
        # 将各自层的loss 相加 得到 内容总cost， 使用的是均方差计算loss
    for i, name in enumerate(style_layers):
        style_loss += compute_style_loss(style_features[name], target_features[name])
        # 将各自层的loss 相加 得到 风格总cost， 先计算目标和风格的gram matrix，再使用均方差计算loss
    loss = content_loss + style_loss
    return loss


def compute_content_loss(content_features, target_features):
    _, height, width, channel = map(lambda i: i.value, target_features.get_shape())
    size = height*width*channel
    return tf.nn.l2_loss(target_features - content_features)/size


def compute_style_loss(style_features, target_features):
    _, height, width, channel = map(lambda i: i.value, target_features.get_shape())
    size = height*width*channel

    # 这里的reshape操作先将 特征图 变为两维， （height*width，channel），再计算 gram matrix
    style_features = tf.reshape(style_features, (-1, channel))
    style_gram = tf.matmul(tf.transpose(style_features), style_features)/size

    target_features = tf.reshape(target_features, (-1, channel))
    target_gram = tf.matmul(tf.transpose(target_features), target_features) / size

    return tf.nn.l2_loss(target_gram - style_gram)/size


if __name__ == '__main__':
    stylize(content_path, style_path)

