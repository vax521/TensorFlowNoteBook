# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:27:22 2018

@author: xingxf03
"""

import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.,2.]])

result = tf.matmul(matrix1,matrix2)
print(result)