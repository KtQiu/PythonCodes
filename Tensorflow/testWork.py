# This file is to check whethe the encironment is right or not.
# And this is my first time to use Anaconda
# author : Kt Qiu
# time :2017-9-11 22:24 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# the first test code
a = tf.random_normal([2,20]) # define a 2x20 martix of random numbers
sess = tf.Session() # start a tf session
out = sess.run(a)
x, y = out # split up the 2x20 martix into two 1x10 vectors ==> x and y
plt.scatter(x,y)
plt.show()


# the second test code
# A = tf.random_normal((10, 2)) # A is a 10x2 martix of random numbers
# B = tf.random_normal((2, 5)) # B is a 2x5 martix of random numbers
# C = tf.matmul(A, B) # matrix multiply A x B
# sess = tf.InteractiveSession()
# print(sess.run(c))
