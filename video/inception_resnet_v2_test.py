#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: qiujiarong
# Date: 27/01/2018
import tensorflow as tf

slim = tf.contrib.slim
from PIL import Image
from video.inception_resnet_v2 import *
import numpy as np
from tensorflow.python import pywrap_tensorflow
from data_provider.example import Example
dir_name='../data/'
checkpoint_file = dir_name+'checkpoints/inception_resnet_v2_2016_08_30.ckpt'
sample_images = ['dog.jpg', 'panda.jpg']
input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
sess = tf.Session()
arg_scope = inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
    logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)

def get_inception_drop_out_value(images):
    ret=[]
    for image in images:
        im = Image.open(image).resize((299, 299))
        im = np.array(im)
        im = im.reshape(-1, 299, 299, 3)
        im = 2 * (im / 255.0) - 1.0
        drop_out_values = sess.run(end_points['PreLogitsFlatten'], feed_dict={input_tensor: im})
        ret.append(drop_out_values.squeeze())
    return np.array(ret)



def show_tensor():
    # http://blog.csdn.net/u010698086/article/details/77916532
    # 显示打印模型的信息
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        print("shape: "+str(reader.get_tensor(key).shape)+" tensor_name: "+key)
    # print_tensors_in_checkpoint_file(checkpoint_file)

def predict_test():
    # input tensor 299*299*3
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
    # Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
    for image in sample_images:
        im = Image.open(dir_name+image).resize((299, 299))
        im = np.array(im)
        im = im.reshape(-1, 299, 299, 3)
        im = 2 * (im / 255.0) - 1.0
        drop_out_values,predict_values, logit_values = sess.run([end_points['PreLogitsFlatten'],end_points['Predictions'], logits], feed_dict={input_tensor: im})
        print (drop_out_values.shape)
        print (np.max(predict_values), np.max(logit_values))
        print (np.argmax(predict_values), np.argmax(logit_values))

if __name__ == '__main__':
    # predict_test()
    # show_tensor()
    print(get_inception_drop_out_value(['/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00001.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00002.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00003.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00004.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00005.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00006.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00007.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00008.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00009.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00010.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00011.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00012.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00013.jpg', '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01/00014.jpg']).shape)