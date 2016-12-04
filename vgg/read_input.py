#Read input
import tensorflow as tf

import utils

import numpy as np
import os
import sys
import random
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class to_one_hot:
    """Convert list of labels to one-hot encoding"""
    def __init__(self):
        self.l_encoder = LabelEncoder()
        self.oh_encoder = OneHotEncoder()

    def fit_transform(self, label_list):
        int_encoded = self.l_encoder.fit_transform(label_list)
        int_encoded = int_encoded.reshape([-1, 1])
        oh_encoded = self.oh_encoder.fit_transform(int_encoded)
        return oh_encoded.toarray().tolist()

    def transform(self, label_list):
        int_encoded = self.l_encoder.transform(label_list)
        int_encoded = int_encoded.reshape([-1, 1])
        oh_encoded = self.oh_encoder.transform(int_encoded)
        return oh_encoded.toarray().tolist()

class imgnet:
    """
    How to use
    1, init an instance, demo = imgnet()
    2, read data, demo.(path_dataset = "../../big_data/Imagenet_dataset/"), no return
    3, batch = demo.train(50) will return a tuple, batch[0] is image nparray, batch[1] is labels
    4, demo.test_images is test image, demo.test_labels is test labels
    """
    def __init__(self, path = None):
        pass

    def read_data_sets(self, path_dataset = None):

        num_total_images = 0
        num_classes = 0
        num_test_size = 500
        
        if not path_dataset:
            path_dataset = "../../big_data/Imagenet_dataset/"

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

        encoder = to_one_hot()
        dataset_labels = encoder.fit_transform(dataset_labels)

        # load testing image_paths & labels
        # assume each line of labelfile stand for a label
        test_dataset_images = list()
        test_dataset_labels = list()

        # create an index list mapping "test_image_file" to "class_code"
        test_image_file_label_index = dict()

        class_code = []
        for test_label_file in test_paths_labels:
            class_code = list() # binary array for class represent

            class_code = encoder.transform([os.path.splitext(test_label_file)[0].split("/")[-1]])

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
        self.test_images = np.array(test_dataset_images)
        self.test_labels = np.array(test_dataset_labels)
        #lb = preprocessing.LabelBinarizer()
        #test_dataset_labels = lb.fit_transform(test_dataset_labels)
        # reshape for tensor
        self.test_images = self.test_images.reshape((num_test_size, 224, 224, 3))

        self.dataset_images = dataset_images
        self.dataset_labels = dataset_labels

    def next_batch(self, num_batch_size = 50):

        batch_images = list()
        batch_labels = list()
        # a random batch of index
        batch_rand = list()
        for _ in range(num_batch_size):
            rand_chosen_ind = random.choice(dataset_toload)
            batch_rand.append(rand_chosen_ind)

        # construct a batch of training data (images & labels)
        for one_sample in batch_rand:
            # here we load real data
            image_file = utils.load_image(dataset_images[one_sample])
            batch_images.append(image_file)
            #print(image_file.shape, dataset_images[one_sample], "remove loaded:", one_sample)
            batch_labels.append(dataset_labels[one_sample])

        # convert list into array
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # TODO: official codes convert (num_batch_size, length, width, depth) into (num_batch_size, length * width * depth)
        # but our training function is different
        batch_images = batch_images.reshape((num_batch_size, 224, 224, 3))

        return (batch_images, batch_labels)