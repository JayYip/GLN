import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, layer_norm
import numpy as np
import sys
from cln import conv_layer_norm

slim = tf.contrib.slim

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
train_mode = tf.placeholder(tf.bool)
batch_size = tf.shape(x)[0]
#if train_mode is not None:
#    batch_size = 50
#else:
#    batch_size = 10000

mode = sys.argv[1]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def bp_conv(X, W, b, output_shape):
    X += b 
    return tf.nn.conv2d_transpose(X, W, output_shape, strides=[1, 1, 1, 1])

def unpooling(X, name='unpool'):
    sh = X.get_shape().as_list()
    dim = len(sh[1:-1])
    tmp = [-1] + sh[-dim:]
    out = (tf.reshape(X, tmp))
    for i in range(dim, 0, -1):
        out = tf.concat(i, [out, tf.zeros_like(out)])
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def bp_fc_2_pool(self, X, height=7, width=7, depth=512):
    return tf.reshape(X, [1, height, width, depth])

def bp_fc(X, W, b):
    X += b 
    return tf.matmul(X, W)

#####################################

if mode == 'cln':
    #y_tr = tf.ones([10])
    pool2_tr = tf.ones([batch_size, 7, 7, 64])
    conv2_tr = unpooling(pool2_tr)
    W_conv2_T = tf.ones([5, 5, 32, 64])
    pool1_tr = bp_conv(conv2_tr, W_conv2_T, tf.zeros([64]), [batch_size, 14, 14, 32])
    pool1_tr.set_shape([None, 14, 14, 32])
    #print (conv2_tr.get_shape().as_list())
    conv1_tr = unpooling(pool1_tr) # bs * 28 * 28 * 32

#####################################

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1]) # 28 * 28

input1 = conv2d(x_image, W_conv1) + b_conv1
if mode == 'bn':
    input1 = batch_norm(inputs=input1, scale=True)
elif mode == 'ln':
    input1 = layer_norm(input1)
elif mode == 'cln':
    input1 = conv_layer_norm(input1, conv1_tr)

h_conv1 = tf.nn.relu(input1)
h_pool1 = max_pool_2x2(h_conv1) #  14 * 14

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

input2 = conv2d(h_pool1, W_conv2) + b_conv2
if mode == 'bn':
    input2 = batch_norm(inputs=input2, scale=True)
elif mode == 'ln':
    input2 = layer_norm(input2)
elif mode == 'cln':
    input2 = conv_layer_norm(input2, conv2_tr)

h_conv2 = tf.nn.relu(input2) # 14 * 14
h_pool2 = max_pool_2x2(h_conv2) # 7 * 7
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
input3 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
if mode == 'bn':
    input3 = batch_norm(inputs=input3, scale=True)
elif mode == 'ln':
    input3 = layer_norm(input3)
elif mode == 'cln':
    input3 = layer_norm(input3)
    pass
h_fc1 = tf.nn.relu(input3)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

input4 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
if mode == 'bn':
    input4 = batch_norm(inputs=input4, scale=True)
elif mode == 'ln':
    input4 = layer_norm(input4)
elif mode == 'cln':
    input4 = layer_norm(input4)
y_conv = tf.nn.softmax(input4)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
if mode == 'bn':
    if update_ops: 
        updates = tf.group(*update_ops)
    with tf.control_dependencies([updates]):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
elif mode == 'ln' or mode == 'cln':
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
new_cn_val = -np.inf
for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        cn_val = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], train_mode: True, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, cn_val))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], train_mode: True, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, train_mode: False, keep_prob: 1.0}))