from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
import os
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=1000, help='Number of steps to run trainer.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')

FLAGS, unparsed = parser.parse_known_args()

num_epoch = FLAGS.num_epoch
learning_rate = FLAGS.learning_rate

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_Classifier(x):
    x_image = x
    
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,3,64],stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1],padding='SAME')+b_conv1)
    
    h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,64,64],stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
    
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
    
    
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)
    
    
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5,strides=[1,1,1,1],padding='SAME')+b_conv5)
    
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8*8*128, 384],stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1,shape=[384]))
    h_conv5_flat = tf.reshape(h_conv5,[-1,8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat,W_fc1)+b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10],stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)
    
    return y_pred, logits

x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)


(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train,10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test,10),axis=1)

y_pred, logits = build_CNN_Classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('test_input', 1)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    
    for i in range(num_epoch):
        batch = next_batch(128, x_train, y_train_one_hot.eval())
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            
            print('Epoch: %d, Accuracy: %f, Loss: %f'%(i, train_accuracy, loss_print))
            
        summary,_ = sess.run([merged,train_step], feed_dict={x: batch[0], y:batch[1], keep_prob:0.8})
        train_writer.add_summary(summary, i)

    train_writer.close()

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000,x_test,y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y:test_batch[1], keep_prob: 1.0})
            
    test_accuracy = test_accuracy/10
    print('Test Accuracy: %f'%(test_accuracy))