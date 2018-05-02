#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-10

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from video.c3d_model import C3DModel
from video.video_provider import C3D
from data_provider.THUMOS14 import THUMOS14

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
from video.utils import show_tensor
num_frames_per_clip=16
crop_size=112
np_mean = np.load(os.path.join("../models/", 'crop_mean.npy')).reshape(
    [num_frames_per_clip, crop_size, crop_size, 3])

batch_size=16
def read_clip(frames):
    def get_frames_data(frames):
        ''' Given a directory containing extracted frames, return a video clip of
        (num_frames_per_clip) consecutive frames as a list of np arrays '''
        ret_arr = []
        for image_name in frames:
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
        return ret_arr

    assert len(frames) == num_frames_per_clip
    tmp_data = get_frames_data(frames)
    data = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if (img.width > img.height):
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(
                np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(
                np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean[j]
        data.append(img)
    np_arr_data = np.array(data).astype(np.float32)
    return np_arr_data

def get_batch(idx,X,y,batch_size=16):
    batch_X,batch_y=[],[]
    cnt=0
    while cnt<batch_size:
        if idx>=len(X):idx=0
        if len(X[idx])<num_frames_per_clip:
            idx+=1
            continue
        s_index = random.randint(0, len(X[idx]) - num_frames_per_clip)
        batch_X.append(read_clip(X[idx][s_index:s_index+num_frames_per_clip]))
        batch_y.append(y[idx])
        idx+=1
        cnt+=1
    return np.array(batch_X),np.array(batch_y),idx
    

def test(model_name):
    dataset = THUMOS14()
    train, test = dataset.load_finetune_info()
    train_X = []
    train_y = []
    for vid, start, end, label in train:
        train_X.append(dataset.get_frames(vid, start, end))
        train_y.append(label)
    test_X = []
    test_y = []
    for vid, start, end, label in test:
        test_X.append(dataset.get_frames(vid,start,end))
        test_y.append(label)
    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)
    print(len(train_X), len(test_X))
    model = C3DModel(num_classes=20, batch_size=16)
    idx = 0

    images_placeholder, labels_placeholder = model.placeholder_inputs()
    print(images_placeholder, labels_placeholder)
    model_name = os.path.join(dataset.MODEL_DIR, )
    print(show_tensor(model_name))

    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = model.placeholder_inputs()
    logits = model.inference_c3d(images_placeholder)[0]
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    # And then after everything is built, start the training loop.

    next_start_pos = 0
    for step in xrange(10):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        start_time = time.time()
        batch_X, batch_y = get_batch(idx, train_X, train_y)
        _loss, _ = sess.run([loss], feed_dict={images_placeholder: batch_X, labels_placeholder: batch_y})
        print(_loss)
    print("done")

def finetune():
    dataset = THUMOS14()
    train, test = dataset.load_finetune_info()
    train_X = []
    train_y = []
    cnt=0
    for vid, start, end, label in train:
        if len(dataset.get_frames(vid, start, end))<num_frames_per_clip:
            cnt+=1
            continue
        train_X.append(dataset.get_frames(vid, start, end))
        train_y.append(label)

    test_X = []
    test_y = []
    # # for vid, start, end, label in test:
    # #     test_X.append(dataset.get_frames(vid,start,end))
    # #     test_y.append(label)
    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)
    print(len(train_X), len(test_X))
    model = C3DModel(num_classes=20, batch_size=batch_size)
    idx = 0

    images_placeholder, labels_placeholder = model.placeholder_inputs()
    print(images_placeholder, labels_placeholder)
    model_name = os.path.join(dataset.MODEL_DIR, "c3d_ucf101_finetune_whole_iter_20000_TF.model")
    print(show_tensor(model_name))

    all_vars = tf.all_variables()
    var_to_restore = []
    for var in all_vars:
        name = var.name.split('/')[1].split(':')[0]
        if name == 'wout' or name == 'bout': continue
        var_to_restore.append(var)
    # x for x in all_vars if not x.name.endswith('wout') or not x.name.endswith('bout')]
    print(var_to_restore)
    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = model.placeholder_inputs()
    logits = model.inference_c3d(images_placeholder)[0]
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)
    saver = tf.train.Saver(var_list=var_to_restore)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, model_name)
    saver2 = tf.train.Saver()
    # And then after everything is built, start the training loop.

    next_start_pos = 0
    for step in xrange(10000):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        batch_X, batch_y,idx= get_batch(idx, train_X, train_y,batch_size=batch_size)
        _loss, _ = sess.run([loss, opt], feed_dict={images_placeholder: batch_X, labels_placeholder: batch_y})
        if step%10==0:
            print(idx,_loss)
        # print(_loss)
    saver2.save(sess, os.path.join(dataset.MODEL_DIR, "c3d_thumos_finetune_whole_iter_1000_TF.model"))
    print("done")

if __name__ == '__main__':
   # model_name="c3d_ucf101_finetune_whole_iter_20000_TF.model"
   # test(model_name)
   # finetune()
   dataset = THUMOS14()
   show_tensor(os.path.join(dataset.MODEL_DIR, "c3d_thumos_finetune_whole_iter_1000_TF.model"))
