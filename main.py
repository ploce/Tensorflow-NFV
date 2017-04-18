#coding=utf-8
import copy
import numpy as np
import tensorflow as tf


FILE_NAME = 'testdata'
BATCH_SIZE = 100

#输入维度
INPUT_DIM_1 = 24
INPUT_DIM_2 = 24
INPUT_DIM = INPUT_DIM_1 * INPUT_DIM_2 + 3 

#输出维度
OUTPUT_DIM = 1

#神经网络维度
NN_DIM = 50

LEARNINT_RATE = 0.01


def process_input():
    return output


def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 1))

def bais_variable(shape):
    return tf.Variable(tf.constant(0., shape = shape))

def main(_):
    x = tf.placeholder(tf.float32, [None, INPUT_DIM], name='input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_DIM], name = 'output')

    with tf.name_scope('policy_network'):
        W1 = weight_variable([INPUT_DIM, NN_DIM])
        b1 = bais_variable([NN_DIM])
        network1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = weight_variable([NN_DIM, NN_DIM])
        b2 = bais_variable([NN_DIM])
        network2 = tf.nn.relu(tf.matmul(network1, W2) + b2)

        W3 = weight_variable([NN_DIM, OUTPUT])
        b3 = bais_variable([NN_DIM])
        y = tf.nn.softmax(tf.matmul(network2, W3) + b3)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(LEARNINT_RATE).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()
    sess.run(init)

    with tf.name_scope('train'):
        for i in range(1000):
            batch_x, batch_y = process_input()
            acc,_ = secc.run([accuracy, train_step], feed_dict={x : batch_x, y_: batch_y})
            print("Accuracy at step %s : %s" % (i, acc))

    with tf.name_scope('test'):
        '''测试训练结果'''

if __name__ == "__main__":
    tf.app.run()
