#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/9/15 14:18
# @Author   : Kt Qiu
# @File     : MNIST_softmax.py
# @Software : PyCharm
# @Email    : kitty666ball@gmail.com
# @WeChat   : helloqkt

import tensorflow as tf
import numpy as py
import input_data

mnist = input_data.read_data_sets("NMIST_data/", one_hot=True)

# init the Variable and Data
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])# "None" means the length can be any number; "784" is 28x28
yActual = tf.placeholder("float", [None, 10])

# the result and the cost function
yPredict = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(yActual*tf.log(yPredict))
# optimize the parameters
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init = tf.initialize_all_variables()
# WARNING:tensorflow:From C:\python36\lib\site-packages\tensorflow\python\util\tf_should_use.py:175:
# initialize_all_variables (from tensorflow.python.ops.variables)
# is deprecated and will be removed after 2017-03-02.
# Instructions for updating: Use `tf.global_variables_initializer` instead.
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# stochastic training
# Using all the data to train costs too much,so I just choose a part of the data.
# What's more, using stochastic training can make best use of the data.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, yActual :batch_ys})

# analysis the result and evaluate my model

correct_predict = tf.equal(tf.argmax(yActual,1),tf.argmax(yPredict,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, yActual:mnist.test.labels}))
