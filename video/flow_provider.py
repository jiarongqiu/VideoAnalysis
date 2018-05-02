#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/04/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

from video.flow_test import dense_optical_flow,dense_optical_flow_test
from data_provider.example import Example
from data_provider.UCF101 import UCF101
from video.rgb_provider import InceptionResNetV2
DIR_NAME = "../data/"

model=InceptionResNetV2()
class Flow(object):
    """"""

    def __init__(self ):
        """Constructor for FlowFeature"""

    # def images2feature(self,images):
    #     return flow_test.dense_optical_flow(images)

    def preprocess(self,frames,out):
        flows=dense_optical_flow(frames)
        flows.dump(out)

    def flow_test(self,frames,out_folder):
        dense_optical_flow_test(frames,out_folder)

    def test(self):
        sample_images = []
        print(self.images2feature(images=sample_images).shape)

    def mean_flow(self):
        FEATURE_DIR = os.path.join(UCF101.FEATURE_DIR, 'flow')
        dataset = UCF101()
        dataset.load_in()
        test_list = dataset.test_list
        train_list = dataset.train_list
        for video, label in train_list:
            print(video)
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            folder=os.path.join(UCF101.FLOW_DIR,name)
            frames = dataset.get_frames(folder)
            feature = model.images2feature(frames, mean=True)
            feature.dump(path)

        for video, label in test_list:
            print(video)
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            folder = os.path.join(UCF101.FLOW_DIR, name)
            frames = dataset.get_frames(folder)
            feature = model.images2feature(frames, mean=True)
            feature.dump(path)


def main():

    # MODEL_DIR = Example.MODEL_DIR
    # EXAMPLE_DIR = Example.EXAMPLE_DIR
    # example = Example()
    flow=Flow()

    flow.mean_flow()
    # flow.preprocess(example.get_frames(),os.path.join(EXAMPLE_DIR,"flow",example.get_video_name()))
    # flow.flow_test(example.get_frames(),os.path.join(EXAMPLE_DIR,"flow"))
    # flow.test()

if __name__ == '__main__':
    main()