#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 31/03/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow.python import pywrap_tensorflow
import os

class Path(object):
    """
        path for saving and loading dataset and models
    """
    model2checkpoints={
        "C3D":"c3d_ucf101_finetune_whoel_iter_20000_TF.model",
        "InceptionResNetV2":"inception_resnet_v2_2016_08_30.ckpt",

    }
    model_dir="../data/checkpoints"
    def __init__(self,model,dataset):
        """Constructor for Path"""






def show_tensor(checkpoint_file):
    # http://blog.csdn.net/u010698086/article/details/77916532
    # 显示打印模型的信息
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print("shape: "+str(reader.get_tensor(key).shape)+" tensor_name: "+key)

def main():
    show_tensor()

if __name__ == '__main__':
    main()