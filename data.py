#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy
from six.moves import urllib
import tensorflow as tf

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000

# Options
# one_hot = put the labels in one hot shape
# default is data for 1 pic in a 1D array

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images, shape = [] ):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, *shape)
    return data

def extract_labels(filename, num_images, shape = [] ):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        labels = labels.reshape(num_images, *shape)
    return labels


# Extract it into numpy arrays.

class Prepare_dataset:

    def __init__(self, batch_size = 32, shape = '1D', dtype='train'):
        self.epoch = 0
        self.train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
        self.train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        self.test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
        self.test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

        self.reshape = [IMAGE_SIZE * IMAGE_SIZE] if (shape == '1D') else [IMAGE_SIZE, IMAGE_SIZE]
        self.train_data = extract_data(self.train_data_filename, 60000, shape = self.reshape)
        self.train_labels = extract_labels(self.train_labels_filename, 60000, shape = [ ])
        self.test_data = extract_data(self.test_data_filename, 10000, shape = self.reshape)
        self.test_labels = extract_labels(self.test_labels_filename, 10000, shape = [ ])

        self.shuffleAll()

        self.validation_data = self.train_data[:VALIDATION_SIZE, ...]
        self.validation_labels = self.train_labels[:VALIDATION_SIZE]
        self.train_data = self.train_data[VALIDATION_SIZE:, ...]
        self.train_labels = self.train_labels[VALIDATION_SIZE:]

        self.train_size = self.train_labels.shape[0]
        self.image_size = self.train_data.shape[1:]
        self.test_size = self.test_data.shape[0]

        self.batch_size = batch_size
        self.step = 0
        self.type = dtype

        # Find how many values are missing to complete an other batch and select them randomply and create a new set
        # missing = ((train_size - train_size%BATCH_SIZE)//BATCH_SIZE + 1)*BATCH_SIZE - train_size
        # tf.random_uniform( [missing], minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

    def shuffleAll(self):
        perm_train = numpy.arange(self.train_data.shape[0])
        numpy.random.shuffle(perm_train)

        self.train_data = self.train_data[perm_train]
        self.train_labels = self.train_labels[perm_train]

        perm_test = numpy.arange(self.test_data.shape[0])
        numpy.random.shuffle(perm_test)

        self.test_data = self.test_data[perm_test]
        self.test_labels = self.test_labels[perm_test]


    def __call__(self, step = 0, dtype = 'train', epoch = 0):
        if (self.epoch != epoch):
            self.shuffleAll()
            self.epoch = epoch

        # shuffle = numpy.arange(self.batch_size)
        # numpy.random.shuffle(shuffle)

        if (dtype=='train'):
            data, labels, size = self.train_data, self.train_labels, self.train_size
            if (self.type != dtype):
                self.type = dtype
                self.step = 0
        else:
            data, labels, size = self.test_data, self.test_labels, self.test_size

        self.step += 1

        offset = (self.step * self.batch_size) % (self.train_size - self.batch_size)
        data = self.train_data[offset:(offset + self.batch_size), ...]
        labels = self.train_labels[offset:(offset + self.batch_size)]

        return [data, labels]

# # Example of curried function: a function that returns an other function.
# def feed_dict_gen(BATCH_SIZE, train_labels_node, train_data_node):
#   def feed_dict_gen(step, kp, type='train'):
#     """format data to feed to session"""
#     if (type=='train'):
#       data, labels, size = train_data, train_labels, train_size
#     else:
#       data, labels, size = test_data, test_labels, test_size

#     offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
#     batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
#     batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    
#     feed_dict = {
#       train_data_node: batch_data,
#       train_labels_node: batch_labels,
#       'keep_prob:0': kp,
#     }
#     return [feed_dict, batch_labels, batch_data]
#   return feed_dict_gen

