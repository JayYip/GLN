import tensorflow as tf

import vgg19
import utils

import numpy as np
import os
import sys
import random
from sklearn import preprocessing

# configuration
# currently i assume the ImageNet dataset consists of tons of image files and a single corresponding label file
num_total_images = 0
num_classes = 0
num_batch_size = 1
num_test_size = 500
path_dataset = "../../big_data/Imagenet_dataset/"
# path_dataset = "dataset/ImageNet/"
learning_rate = 0.001
mode = sys.argv[1]
# load training image_path & labels
# at this stage, just load filename rather than real data
dataset_images = list()
dataset_labels = list() # for test
test_paths_labels = list()
for subdir in os.listdir(path_dataset):
    if subdir.startswith('.') or "test" == subdir:
        continue
    elif os.path.isfile(path_dataset + subdir):
        test_paths_labels.append(path_dataset + subdir)
        continue
    for image_file_name in os.listdir(path_dataset + subdir):
        if image_file_name.startswith('.'):
            continue
        image = path_dataset + subdir + '/' + image_file_name
        dataset_images.append(image)
        dataset_labels.append(subdir) # for test
        num_total_images += 1
num_classes = len(set(dataset_labels)) # for test
text_classes = list(set(dataset_labels)) # for test
for cls_i in range(len(text_classes)): # for test
    for i in range(len(dataset_labels)):
        if dataset_labels[i] == text_classes[cls_i]:
            dataset_labels[i] = [1 if j == cls_i else 0 for j in range(num_classes)]
# generate synset.txt (labels' text for printing)
with open("./synset.txt", "w") as f:
    for cls in text_classes:
        f.write(cls + "\n")

# load testing image_paths & labels
# assume each line of labelfile stand for a label
test_dataset_images = list()
test_dataset_labels = list()

# create an index list mapping "test_image_file" to "class_code"
test_image_file_label_index = dict()
for test_label_file in test_paths_labels:
    class_code = list() # binary array for class represent
    for cls_i in range(len(text_classes)):
        if os.path.splitext(test_label_file)[0].split("/")[-1] == text_classes[cls_i]:
            class_code = [1 if j == cls_i else 0 for j in range(num_classes)]
    with open(test_label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            test_image_file_name = line.strip("\n")
            #print(test_image_file_name)
            test_image_file_label_index[test_image_file_name] = class_code
            # print(test_image_file_name, class_code)
# load all test images
#print (test_image_file_label_index)
for test_image_file_name in os.listdir(path_dataset + "test"):
    if test_image_file_name.startswith('.'):
        continue
    test_image = utils.load_image(path_dataset + "test/" + test_image_file_name)
    test_dataset_images.append(test_image)
    #print("load test image:", test_image.shape, test_image_file_name)
    #print(test_image_file_name, test_image_file_label_index[test_image_file_name])
    test_dataset_labels.append(test_image_file_label_index[test_image_file_name])
# convert list into array
#print (test_dataset_labels)
test_dataset_images = np.array(test_dataset_images)
test_dataset_labels = np.array(test_dataset_labels)
#lb = preprocessing.LabelBinarizer()
#test_dataset_labels = lb.fit_transform(test_dataset_labels)
# reshape for tensor
test_dataset_images = test_dataset_images.reshape((num_test_size, 224, 224, 3))

if __name__=='__main__':

    sess = tf.InteractiveSession()

    images = tf.placeholder(tf.float32, [num_batch_size, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [num_batch_size, num_classes]) # totally num_classes categories
    # test dataset
    train_mode = tf.placeholder(tf.bool)

    #vgg = vgg19.Vgg19('./vgg19.npy')
    print (mode)
    vgg = vgg19.Vgg19(num_batch_size, norm_mode=mode)
    #vgg.get_tr()
    vgg.build_net(images, train_mode)
    cost = tf.reduce_sum((vgg.prob - labels) ** 2) 
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(vgg.prob), reduction_indices=[1]))
    #cost_val = cost.eval(feed_dict=train_feed_dict)
    #print ('cross entropy: ', cost_val)
    train = tf.train.AdamOptimizer(0.0001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    # print(vgg.get_var_count())

    #Change here!!!!!!!!!!!
    g=tf.get_default_graph()
    p1=tdb.plot_op(viz.viz_conv_weights,inputs=[g.as_graph_element(tr_conv5_4)])
    p2=tdb.plot_op(viz.viz_conv_weights,inputs=[g.as_graph_element(tr_conv5_3)])
    p3=tdb.plot_op(viz.viz_conv_weights,inputs=[g.as_graph_element(tr_conv5_2)])
    p4=tdb.plot_op(viz.viz_conv_weights,inputs=[g.as_graph_element(tr_conv5_1)])


    sess.run(tf.initialize_all_variables())

    num_data_trained = 0
    # for loading images randomly
    dataset_toload = [i for i in range(len(dataset_images))]
    print("check", len(dataset_toload), len(dataset_images))
    random.seed()
    for i in range(1, 10000):
        # a batch of data
        print ('iteration:', i)
        batch_images = list()
        batch_labels = list()
        # a random batch of index
        batch_rand = list()
        for _ in range(num_batch_size):
            rand_chosen_ind = 0#random.choice(dataset_toload)
            batch_rand.append(rand_chosen_ind)
            #dataset_toload.remove(rand_chosen_ind)

        # construct a batch of training data (images & labels)
        for one_sample in batch_rand:
            # here we load real data
            image_file = utils.load_image(dataset_images[one_sample])
            batch_images.append(image_file)
            print(image_file.shape, dataset_images[one_sample], "remove loaded:", one_sample)
            batch_labels.append(dataset_labels[one_sample])

        # convert list into array
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # TODO: official codes convert (num_batch_size, length, width, depth) into (num_batch_size, length * width * depth)
        # but our training function is different
        batch_images = batch_images.reshape((num_batch_size, 224, 224, 3))
        
        train_feed_dict = {
            images : batch_images,
            labels : batch_labels,
            train_mode : True
        }

        # simple 1-step training, train with one image
        cost_val = cost.eval(feed_dict=train_feed_dict)
        print ('cross entropy: ', cost_val)
        sess.run(train, feed_dict=train_feed_dict)
        print (vgg.prob.eval(feed_dict=train_feed_dict))
        if i % 10 == 0:
            with open('./cost.txt', 'a') as f:
                f.write(str(cost_val)+'\n')
        #for i in range(10):
        #    utils.print_prob(pred[i], './synset.txt')
        

        #Change plots here!!!!!!!!!!!!!!!!!!!!!
        if step % 10 == 0:  
            status,result=tdb.debug([p1,p2,p3,p4], feed_dict=train_feed_dict, breakpoints=None, break_immediately=False, session=sess)


        if i % 111111111 == 0:
            #train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
            acc_sum = 0#accuracy.eval(feed_dict=test_feed_dict)
            for num in range(50):
                test_feed_dict = {
                    images : test_dataset_images[num_batch_size*num:num_batch_size*num+num_batch_size], 
                    labels : test_dataset_labels[num_batch_size*num:num_batch_size*num+num_batch_size], 
                    train_mode: False
                }
                
                acc_sum += accuracy.eval(feed_dict=test_feed_dict)
                #print (num, acc)
                print(acc_sum)
            acc_sum /= 50
            with open('./ln_accuracy.txt', 'a') as f:
                f.write(str(acc_sum)+'\n')

        #if i % 50 == 0：

    # test save
    #vgg.save_npy(sess, './test-save.npy')