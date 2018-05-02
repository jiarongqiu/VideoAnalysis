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
from data_provider.example import Example
from data_provider.UCF101 import UCF101


MODEL_DIR = Example.MODEL_DIR
EXAMPLE_DIR = Example.EXAMPLE_DIR


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
            out, end_points = inception_resnet_v2(self.input_tensor, is_training=False,dropout_keep_prob=1)
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)
        self.end_points = end_points
        self.sess = sess
        self.out=out
        id2label={}
        with open(os.path.join(MODEL_DIR,"image_net_label.txt"),'r') as fr:
            for i,line in enumerate(fr):
                id2label[i]=line
        self.id2label=id2label
        print("Inception Model Load In Finished")

    def preprocess(self, image_file, crop_size=299):
        im = Image.open(image_file)
        im = im.resize((crop_size, crop_size))
        im = np.array(im)
        im = im.reshape(-1, 299, 299, 3)
        im = 2 * (im / 255.0) - 1.0
        return im

    # possible to enhance
    def images2feature(self, images,method='logit',mean=False):
        ret = []
        for image in images:
            im = self.preprocess(image)
            if method =='logit':
                drop_out_values = self.sess.run(self.end_points['Logits'], feed_dict={self.input_tensor: im})
            ret.append(drop_out_values.squeeze())
        ret=np.array(ret)
        if mean:
            ret=np.mean(ret,axis=0)
        return ret

    def get_label(self,idx):
        return self.id2label[idx]

    def test(self,images):
        for image in images:
            im = self.preprocess(image)
            logit = self.sess.run(self.out, feed_dict={self.input_tensor: im})
            print(logit.shape)
            idx=np.argmax(logit)
            print("result:", self.get_label(idx), "score:", logit[0,idx])
        # print(self.images2feature(sample_images).shape)
        # print("Inception Test Finished")

    def UCF101_mean_feature(self):
        FEATURE_DIR = os.path.join(UCF101.FEATURE_DIR, 'rgb_logit')
        dataset = UCF101()
        dataset.load_in()
        test_list = dataset.test_list
        train_list = dataset.train_list
        for video, label in train_list:
            print(video)
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            frames=dataset.get_frames(video)
            feature = self.images2feature(frames,mean=True)
            feature.dump(path)
        for video, label in test_list:
            print(video)
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            frames = dataset.get_frames(video)
            feature = self.images2feature(frames,mean=True)
            feature.dump(path)

def main():
    inception = InceptionResNetV2()
    # example = Example()
    # inception.test(example.get_frames(num=1))
    # show_tensor(inception.checkpoint_file)
    inception.UCF101_mean_feature()

if __name__ == '__main__':
    main()
