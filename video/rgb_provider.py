#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 31/03/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import os
import numpy as np
from PIL import Image

from video.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from video.utils import show_tensor
DIR_NAME = "../data/"
MODEL_DIR = os.path.join(DIR_NAME, "checkpoints")
DEMO_DIR = os.path.join(DIR_NAME, "demo")

slim = tf.contrib.slim


class InceptionResNetV2(object):
    """
        Inception ResNet V2 pretrained on ImageNet
    """

    def __init__(self, ):
        """Constructor for InceptionResNetV2"""
        self.checkpoint_file = os.path.join(MODEL_DIR, 'inception_resnet_v2_2016_08_30.ckpt')
        self.input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
        sess = tf.Session()
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = inception_resnet_v2(self.input_tensor, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)
        self.end_points = end_points
        self.sess = sess
        print("Inception Model Load In Finished")

    def preprocess(self, image_file, crop_size=299):
        im = Image.open(image_file)
        im = im.resize((crop_size, crop_size))
        im = np.array(im)
        im = im.reshape(-1, 299, 299, 3)
        im = 2 * (im / 255.0) - 1.0
        return im

    # possible to enhance
    def images2feature(self, images):
        ret = []
        for image in images:
            im = self.preprocess(image)
            drop_out_values = self.sess.run(self.end_points['PreLogitsFlatten'], feed_dict={self.input_tensor: im})
            ret.append(drop_out_values.squeeze())
        return np.array(ret)

    def test(self):
        sample_images = [os.path.join(DEMO_DIR, 'dog.jpg')]
        print(self.images2feature(sample_images).shape)
        print("Inception Test Finished")


def main():
    inception = InceptionResNetV2()
    inception.test()
    show_tensor(inception.checkpoint_file)


if __name__ == '__main__':
    main()
