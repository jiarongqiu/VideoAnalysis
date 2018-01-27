# http://blog.csdn.net/u011961856/article/details/77064631
# coding:utf-8
# tensorflow模型保存文件分析
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow

# http://blog.csdn.net/u010698086/article/details/77916532
# 显示打印模型的信息
model_dir = "../models/"
checkpoint_path = os.path.join(model_dir, "inception_resnet_v2_2016_08_30.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names
