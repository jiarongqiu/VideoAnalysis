#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 27/01/2018
import tensorflow as tf

slim = tf.contrib.slim
from PIL import Image
from inception_resnet_v2 import *
import numpy as np

# input tensor 299*299*3
input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'
sample_images = ['dog.jpg', 'panda.jpg']
# Load the model
sess = tf.Session()
arg_scope = inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
    logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
for image in sample_images:
    im = Image.open(image).resize((299, 299))
    im = np.array(im)
    im = im.reshape(-1, 299, 299, 3)
    im = 2 * (im / 255.0) - 1.0
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
    print (np.max(predict_values), np.max(logit_values))
    print (np.argmax(predict_values), np.argmax(logit_values))
