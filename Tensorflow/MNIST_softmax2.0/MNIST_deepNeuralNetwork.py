#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/9/16 9:35
# @Author   : Kt Qiu
# @File     : MNIST_deepNeuralNetwork.py
# @Software : PyCharm
# @Email    : kitty666ball@gmail.com
# @WeChat   : helloqkt

import input_data
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    # generate a truncated normal distribution, and the standard devition is 0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
# the difference between Session() and InteractiveSession:
# if you use Session(), you have to implement the whole graph before launching the graph
# and the InteractiveSession() give you the possibility to do this in process.
x = tf.placeholder("float", shape=[None, 784])
yActual = tf.placeholder("float", shape=[None, 10])

# the first layer
W_conv1 = weight_variable([5, 5, 1, 32])
# the first two para [5 5] is the size of the patch size,
# and the last two parameters [1 32] is the input size and output size
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the second layer
W_conv2 = weight_variable([5, 5, 32, 64])  # [5 5 32 64] means 5x5 patch 32 input 64 output
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# set h_pool2 be a vector( None x 1 )
# so that we can use fully connected layer(like file MNIST_softmax 1.0)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout layer: avoid overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(yActual * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(yActual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], yActual: batch[1], keep_prob: 1.0})
        print("step %d, train accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], yActual: batch[1], keep_prob: 0.5})

print("\n\ntrain accuracy:%g" % accuracy.eval(
    feed_dict={x: mnist.test.images, yActual: mnist.test.labels, keep_prob: 1.0}))

# Conclusion:
# A CNN consists of an input and an output layer, as well as multiple hidden layers.
# The hidden layers are either convolutional, pooling or fully connected.
# the fully connrcted layer is usually in the simple MLP
#
