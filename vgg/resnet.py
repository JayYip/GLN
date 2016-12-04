import tensorflow as tf
import resnet_v1
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
pred = resnet_v1.resnet_v1_50(x)

cross_entropy = -tf.reduce_sum(y_ * tf.log(pred))
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
#if mode == 'bn':
#    if update_ops: 
#        updates = tf.group(*update_ops)
#    with tf.control_dependencies([updates]):
#        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#elif mode == 'ln' or mode == 'cln':
#    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
new_cn_val = -np.inf
for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        cn_val = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g" % (i, cn_val))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))