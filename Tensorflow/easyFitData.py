#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/9/14 19:20
# @Author   : Kt Qiu
# @File     : easyFitData.py
# @Software : PyCharm
# @Email    : kitty666ball@gmail.com
# @WeChat   : helloqkt

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

X = np.random.rand(100).astype("float32")
Y = X * 3 + 0.5

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
Ypredict = W * X + b

loss = tf.reduce_mean(tf.square(Ypredict - Y))
# use Gradient descent to minimax the loss
# the learning rate is 0.5
optimzer = tf.train.GradientDescentOptimizer(0.5)
train = optimzer.minimize(loss)

# init the Variables before we 'run' this code every time
init = tf.initialize_all_variables()
# launch the graph
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b), sess.run(loss))
'''
Attention: 
in python3.x, function xrange() does not exist instead range()
usage:  range() 
1. range(int stop)  ==> the start is 0
2. range(int start, int stop, int step)  
'''

'''
result : 

0 [ 1.62885952] [ 1.57638037] 0.355169
20 [ 2.64850068] [ 0.67642069] 0.0109516
40 [ 2.91907668] [ 0.54061615] 0.000580466
60 [ 2.98136926] [ 0.5093509] 3.07673e-05
80 [ 2.99571109] [ 0.50215268] 1.63054e-06
100 [ 2.99901271] [ 0.50049561] 8.63992e-08
120 [ 2.99977279] [ 0.50011408] 4.57577e-09
140 [ 2.99994755] [ 0.50002635] 2.4355e-10
160 [ 2.99998784] [ 0.50000614] 1.29801e-11
180 [ 2.99999714] [ 0.50000143] 7.12603e-13
200 [ 2.99999857] [ 0.50000066] 1.64349e-13

'''

'''
before you active a seesion and function run() is be called,
the python code using tensorflow will not do anything
'''
