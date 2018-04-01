#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/04/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf

from video.utils import show_tensor
from video.c3d_model import placeholder_inputs,inference_c3d
DIR_NAME = "../data/"
MODEL_DIR = os.path.join(DIR_NAME, "checkpoints")
DEMO_DIR = os.path.join(DIR_NAME, "demo")

class C3D(object):
    """"""
    
    def __init__(self, ):
        """Constructor for C3D"""
        self.checkpoint_file = os.path.join(MODEL_DIR, 'c3d_ucf101_finetune_whole_iter_20000_TF.model')
        self.input_tensor,_=placeholder_inputs()
        _,self.end_points=inference_c3d(self.input_tensor)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)
        self.sess = sess
        print(self.end_points['fc_7'])
        print("C3D Model Load In Finished")


def main():
    c3d=C3D()
    # show_tensor(c3d.checkpoint_file)


if __name__ == '__main__':
    main()