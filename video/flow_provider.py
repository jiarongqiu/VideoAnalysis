#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/04/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from video import flow_test

DIR_NAME = "../data/"
MODEL_DIR = os.path.join(DIR_NAME, "checkpoints")
DEMO_DIR = os.path.join(DIR_NAME, "demo")


class FLOW(object):
    """"""

    def __init__(self ):
        """Constructor for FlowFeature"""

    def images2feature(self,images):
        return flow_test.dense_optical_flow(images)

    def test(self):
        sample_images = []
        for i in range(1,6):
            sample_images.append(os.path.join(DEMO_DIR,str(i)+".jpg"))
        print(self.images2feature(images=sample_images).shape)


def main():
    flow=FLOW()
    flow.test()

if __name__ == '__main__':
    main()