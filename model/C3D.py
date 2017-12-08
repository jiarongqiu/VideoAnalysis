# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""

import tensorflow as tf


def init_weight(shape,dev=0.01):
    return tf.Variable(tf.random_normal(shape, stddev=dev))


def conv3d(X, w, b):
    return tf.nn.bias_add(
        tf.nn.conv3d(X, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)


def max_pool(X, k):
    return tf.nn.max_pool3d(X, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME')


class C3D(object):
    """docstring for ClassName"""

    def __init__(self, num_class=11, crop_size=128, frames_per_clip=16):
        super(C3D, self).__init__()
        self.num_class = num_class
        self.crop_size = crop_size
        self.frames_per_clop = frames_per_clip

    def model(self, X, batch_size=16, drop_out=0.5):

        conv1 = conv3d(X, init_weight([3, 3, 3, 3, 64]), tf.zeros([64]))
        conv1 = tf.nn.relu(conv1)
        pool1 = max_pool(conv1, k=1)

        # Convolution Layer
        conv2 = conv3d(pool1, init_weight([3, 3, 3, 64, 128]), tf.zeros([128]))
        conv2 = tf.nn.relu(conv2)
        pool2 = max_pool(conv2, k=2)

        # Convolution Layer
        conv3 = conv3d(pool2, init_weight([3, 3, 3, 128, 256]), tf.zeros([256]))
        conv3 = tf.nn.relu(conv3)
        # conv3 = conv3d(conv3, init_weight([3, 3, 3, 256, 256]), tf.zeros([256]))
        # conv3 = tf.nn.relu(conv3)
        pool3 = max_pool(conv3, k=2)

        # Convolution Layer
        conv4 = conv3d(pool3, init_weight([3, 3, 3, 256, 256]), tf.zeros([256]))
        conv4 = tf.nn.relu(conv4)
        # conv4 = conv3d(conv4, init_weight([3, 3, 3, 512, 256]), tf.zeros([256]))
        # conv4 = tf.nn.relu(conv4)
        pool4 = max_pool(conv4, k=2)

        # Convolution Layer
        conv5 = conv3d(pool4, init_weight([3, 3, 3, 256, 256]), tf.zeros([256]))
        conv5 = tf.nn.relu(conv5)
        # conv5 = conv3d(conv5, init_weight([3, 3, 3, 512, 512]), tf.ones([512])/10)
        # conv5 = tf.nn.relu(conv5)
        pool5 = max_pool(conv5, k=2)

        # Fully connected layer
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        fc6 = tf.reshape(pool5, [batch_size, 4096])  # Reshape conv3 output to fit dense layer input
        fc6 = tf.matmul(fc6, init_weight([4096, 2048],dev=0.005) + tf.ones([2048]))
        fc6 = tf.nn.relu(fc6)  # Relu activation
        fc6 = tf.nn.dropout(fc6, drop_out)

        fc7 = tf.matmul(fc6, init_weight([2048, 2048],dev=0.005) + tf.ones([2048]))
        fc7 = tf.nn.relu(fc7,)  # Relu activation
        fc7 = tf.nn.dropout(fc7, drop_out)

        # Output: class prediction
        fc8 = tf.matmul(fc7, init_weight([2048, self.num_class]) + tf.zeros([self.num_class]))
        fc8 = tf.nn.softmax(fc8)
        return fc8


if __name__ == '__main__':
    images_placeholder = tf.placeholder(tf.float32, shape=(1, 16, 128, 128, 3))
    c3d = C3D()
    print c3d.model(images_placeholder, 1, 0.5)
