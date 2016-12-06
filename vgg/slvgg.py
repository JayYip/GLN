import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm , layer_norm
import numpy as np
import sys
from cln4conv import conv_layer_norm
#from read_input import imgnet
from tensorflow.models.image.cifar10 import cifar10_input
import os


img_sz = 64


#HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def distorted_inputs(batch_size, data_dir= '../../cifardataset/cifar-10-batches-bin'):
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)
  images = tf.image.resize_images(
      tf.cast(images, tf.float32), 
      tf.convert_to_tensor([64,64], dtype=tf.int32))
  labels = tf.one_hot(tf.cast(labels, tf.int32), depth=10, dtype=tf.int32)
  return (images, labels)

def inputs(batch_size, eval_data='test_batch', data_dir = '../../cifardataset/cifar-10-batches-bin'):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)

  images = tf.image.resize_images(
      tf.cast(images, tf.float32), 
      tf.convert_to_tensor([64,64], dtype=tf.int32))
  labels = tf.one_hot(tf.cast(labels, tf.int32), depth=10, dtype=tf.int32)
  return (images, labels)

#END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#x = tf.placeholder("float", shape=[None, img_sz, img_sz, 3])
#y_ = tf.placeholder("float", shape=[None, 5])
#train_mode = tf.placeholder(tf.bool)

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
    X = tf.mul(X, 0.25)
    sh = X.get_shape().as_list()
    dim = len(sh[1:-1])
    tmp = [-1] + sh[-dim:]
    out = (tf.reshape(X, tmp))
    for i in range(dim, 0, -1): # 
        out = tf.concat(i, [out, out])
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def bp_fc_2_pool(self, X, height=7, width=7, depth=512):
    return tf.reshape(X, [1, height, width, depth])

def bp_fc(X, W, b):
    X += b 
    return tf.matmul(X, W)


with tf.Graph().as_default():
  sess = tf.Session()
  x, y_ = distorted_inputs(10)
  batch_size = tf.shape(x)[0]

  #####################################

  def slvgg(x, mode):
      if mode == 'cln':
          y_tr = tf.ones([batch_size, 10])
          
          W_fc2_T = tf.ones([10, 1024]) * (1/10)
          f1_tr = tf.matmul(y_tr, W_fc2_T) * 0.5 

          W_fc1_T = tf.ones([1024, 4*4*256]) * (1/1024)
          pool4_tr_flat = tf.matmul(f1_tr, W_fc1_T) * 0.5
          pool4_tr = tf.reshape(pool4_tr_flat, [batch_size, 4, 4, 256]) 
          conv4_tr = unpooling(pool4_tr) * 0.5 # 8 * 8

          W_conv4_T = tf.ones([5, 5, 128, 256]) * (1/(5*5*256))
          pool3_tr = bp_conv(conv4_tr, W_conv4_T, tf.zeros([256]), [batch_size, 8, 8, 128])
          pool3_tr.set_shape([None, 8, 8, 128])
          conv3_tr = unpooling(pool3_tr) * 0.5 # 16 * 16
          
          W_conv3_T = tf.ones([5, 5, 64, 128]) * (1/(5*5*128))
          pool2_tr = bp_conv(conv3_tr, W_conv3_T, tf.zeros([128]), [batch_size, 16, 16, 64])
          pool2_tr.set_shape([None, 16, 16, 64])
          conv2_tr = unpooling(pool2_tr) * 0.5 # 32 * 32

          W_conv2_T = tf.ones([5, 5, 32, 64]) * (1/(5*5*64))
          pool1_tr = bp_conv(conv2_tr, W_conv2_T, tf.zeros([64]), [batch_size, 32, 32, 32])
          pool1_tr.set_shape([None, 32, 32, 32])
          conv1_tr = unpooling(pool1_tr) # bs * 64 * 64 * 3

      W_conv1 = weight_variable([5, 5, 3, 32])
      b_conv1 = bias_variable([32])
      #x_image = tf.reshape(x, [-1, 28, 28, 1]) 
      # x [-1, 32, 32, 3]
      input1 = conv2d(x, W_conv1) + b_conv1
      if mode == 'bn':
          input1 = batch_norm(inputs=input1, scale=True)
      elif mode == 'ln':
          input1 = layer_norm(input1)
      elif mode == 'cln':
          input1 = conv_layer_norm(input1, conv1_tr, 10)
      h_conv1 = tf.nn.relu(input1)
      h_pool1 = max_pool_2x2(h_conv1) #  32 * 32

      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])

      input2 = conv2d(h_pool1, W_conv2) + b_conv2
      if mode == 'bn':
          input2 = batch_norm(inputs=input2, scale=True)
      elif mode == 'ln':
          input2 = layer_norm(input2)
      elif mode == 'cln':
          input2 = conv_layer_norm(input2, conv2_tr, 10)

      h_conv2 = tf.nn.relu(input2) # 16 * 16
      h_pool2 = max_pool_2x2(h_conv2) # 16 * 16 * 64

      W_conv3 = weight_variable([5, 5, 64, 128])
      b_conv3 = bias_variable([128])

      input3 = conv2d(h_pool2, W_conv3) + b_conv3
      if mode == 'bn':
          input3 = batch_norm(inputs=input3, scale=True)
      elif mode == 'ln':
          input3 = layer_norm(input3)
      elif mode == 'cln':
          input3 = conv_layer_norm(input3, conv3_tr, 10)
      h_conv3 = tf.nn.relu(input3)
      h_pool3 = max_pool_2x2(h_conv3) # 8 * 8 * 64

      W_conv4 = weight_variable([5, 5, 128, 256])
      b_conv4 = bias_variable([256])

      input4 = conv2d(h_pool3, W_conv4)+ b_conv4#tf.matmul(h_conv3_flat, W_fc1) + b_fc1
      if mode == 'bn':
          input4 = batch_norm(inputs=input4, scale=True)
      elif mode == 'ln':
          input4 = layer_norm(input4)
      elif mode == 'cln':
          input4 = conv_layer_norm(input4, conv4_tr, 10)

      h_conv4 = tf.nn.relu(input4) 
      h_pool4 = max_pool_2x2(h_conv4) # 4 * 4 * 128
      h_pool4_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 256])

      W_fc1 = weight_variable([4 * 4 * 256, 1024])
      b_fc1 = bias_variable([1024])

      input4 = tf.matmul(h_pool4_flat, W_fc1) + b_fc1
      h_fc1 = tf.nn.relu(input4)

      keep_prob = 1#tf.placeholder("float")
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])

      input5 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      if mode == 'bn':
          input5 = batch_norm(inputs=input5, scale=True)
      elif mode == 'ln':
          input5 = layer_norm(input5)
      elif mode == 'cln':
          input5 = layer_norm(input5)

      y_conv = tf.nn.softmax(input5)

      return y_conv

  y_conv = slvgg(x, mode)
  y_ = tf.cast(y_, tf.float32)
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
  loss_sum = 0
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
  if mode == 'bn':
      if update_ops: 
          updates = tf.group(*update_ops)
      with tf.control_dependencies([updates]):
          train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  elif mode == 'ln' or mode == 'cln':
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  #Build accuracy on test data
  x_test, y_test = inputs(500)
  y_test_ = slvgg(x_test, mode)
  y_test = tf.cast(y_test, tf.float32)
  correct_prediction = tf.equal(tf.argmax(y_test, 1), tf.argmax(y_test_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)
  sess.run(tf.initialize_all_variables())
  new_cn_val = -np.inf
  for i in range(1, 50001):
      print(str(i))
      #HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      _, loss = sess.run([train_step, cross_entropy])
      #END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if i % 50 == 0:
          with open('loss'+mode+'.txt', 'a') as f1:
              f1.write(str(loss_sum)+'\n')
          print("step %d, training cross_entropy %g" % (i, loss_sum))
          
          test_acc = sess.run(accuracy)
          with open('test_acc_'+mode+'.txt', 'a') as f2:
            f2.write(str(test_acc)+'\n')
          print("test accuracy %g" % sess.run(accuracy))
          #END!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          loss_sum = 0
      else:
          loss_sum += loss

