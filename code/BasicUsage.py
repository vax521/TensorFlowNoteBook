# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:27:22 2018

@author: xingxf03
"""

import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])

matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

#创建会话对象
#sess = tf.Session()
##会话中载入图
#result = sess.run(product)
#
#print(result)
#
#sess.close()
with tf.Session() as sess:
    result = sess.run([product])
    print(result)

#变量维持了图执行过程中的状态信息
state = tf.Variable(0,name="counter")
#Create an op to add one to 'state'
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#初始化Variables
init_op = tf.global_variables_initializer()

#Launch the graph and run the ops
with tf.Session() as sess:
    #Run the 'init' op
    sess.run(init_op)
    #print the initial value of 'state'
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        
#Fetches 取回
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(1.0)
intermed = tf.add(input2,input3)
mul = tf.multiply(input1,intermed)
with tf.Session() as sess:
    result = sess.run([mul,intermed])
    print(result)

#Feed机制       
input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32)  
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))
#Output:[array([ 14.], dtype=float32)]

