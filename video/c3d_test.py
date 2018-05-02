#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/04/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import tensorflow as tf
import video.input_data
from video.c3d_model import C3DModel
import numpy as np
import json
from data_provider.UCF101 import UCF101
from video.video_provider import C3D
from sklearn import svm
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
FLAGS = flags.FLAGS


def end2end():
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    model = C3D()

    pred = []
    y = []
    cnt=0
    for video, label in test_list:
        pred.append(model.predict(dataset.get_random_clip(video)))
        y.append(label)

    write_file = open(os.path.join(dataset.RES_DIR, "c3d_end.txt"), "w")
    for i in range(len(pred)):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, pred[i], 0))
    write_file.close()
def fc7_sgd(l2=False):
    method='c3d_fc7'
    if l2:
        name='c3d_fc7_l2_sgd'
    else:
        name='c3d_fc7_sgd'
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt=0
    for video, label in train_list:
        video=video.split('/')[-1]
        feature=dataset.load_feature(video,method=method,l2=l2)
        X.append(feature)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    clf = SGDClassifier()
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, name), 'w'))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        X.append(dataset.load_feature(video,method=method,l2=l2))
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    # print(clf.predict(X))
    write_file = open(os.path.join(dataset.RES_DIR, name+".txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()


def fc7_svm(l2=False):
    method='c3d_fc7'
    if l2:
        name='c3d_fc7_l2_svm'
    else:
        name='c3d_fc7_svm'
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt=0
    for video, label in train_list:
        video=video.split('/')[-1]
        feature=dataset.load_feature(video,method=method,l2=l2)
        X.append(feature)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, name), 'w'))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        X.append(dataset.load_feature(video,method=method,l2=l2))
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    # print(clf.predict(X))
    write_file = open(os.path.join(dataset.RES_DIR, name+".txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

def rgb_logit(l2=False):
    method='rgb_logit'
    if l2:
        name='rgb_logit_l2_svm'
    else:
        name='rgb_logit_svm'
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>500:break
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # print(y)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, name), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    write_file = open(os.path.join(dataset.RES_DIR, name+".txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

def flow_logit(l2=False):
    method='flow_logit'
    if l2:
        name='flow_logit_l2_svm'
    else:
        name='flow_logit_svm'
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>500:break
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # print(y)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, name), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        # print(video,feature)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    # print(clf.predict(X),y)
    print(X.shape, y.shape)
    write_file = open(os.path.join(dataset.RES_DIR, name+".txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

def audio(l2=False):
    method='vgg'
    if l2:
        name='audio_l2_svm'
    else:
        name='audio_svm'
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>500:break
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # print(y)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, name), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        feature = dataset.load_feature(video, method=method, l2=l2)
        # print(video,feature)
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    # print(clf.predict(X),y)
    print(X.shape, y.shape)
    write_file = open(os.path.join(dataset.RES_DIR, name+".txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

#2002 dims
def two_stream(l2=True):
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        feature=np.hstack((rgb_feature,flow_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>500:break
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # print(y)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    if l2:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'two_stream_l2_svm'), 'w'))
    else:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'two_stream_svm'), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        feature = np.hstack((rgb_feature, flow_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    # print(clf.predict(X),y)
    print(X.shape, y.shape)
    if l2:
        write_file = open(os.path.join(dataset.RES_DIR, "two_stream_l2_svm.txt"), "w")
    else:
        write_file = open(os.path.join(dataset.RES_DIR, "two_stream_svm.txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()


#6098 dims
def visual(l2=True):
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        video_feature =dataset.load_feature(video, method='c3d_fc7', l2=l2)
        feature=np.hstack((rgb_feature,flow_feature,video_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    # print(y)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    if l2:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'visual_l2_svm'), 'w'))
    else:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'visual_svm'), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        video_feature = dataset.load_feature(video, method='c3d_fc7', l2=l2)
        feature = np.hstack((rgb_feature, flow_feature, video_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    # print(clf.predict(X),y)
    print(X.shape, y.shape)
    if l2:
        write_file = open(os.path.join(dataset.RES_DIR, "visual_l2_svm.txt"), "w")
    else:
        write_file = open(os.path.join(dataset.RES_DIR, "visual_svm.txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

#6098 dims
def visual_audio(l2=True):
    dataset = UCF101()
    dataset.load_in()
    test_list = dataset.test_list
    train_list = dataset.train_list
    X = []
    y = []
    cnt = 0
    for video, label in train_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        video_feature =dataset.load_feature(video, method='c3d_fc7', l2=l2)
        audio_feature = dataset.load_feature(video, method='vgg', l2=l2)
        feature=np.hstack((rgb_feature,flow_feature,video_feature,audio_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    if l2:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'visual_audio_l2_svm'), 'w'))
    else:
        pickle.dump(clf, open(os.path.join(UCF101.MODEL_DIR, 'visual_audio_svm'), 'w'))
    # print(clf.predict(X))
    X = []
    y = []
    cnt = 0
    for video, label in test_list:
        video = video.split('/')[-1]
        rgb_feature = dataset.load_feature(video, method='rgb_logit', l2=l2)
        flow_feature = dataset.load_feature(video, method='flow_logit', l2=l2)
        video_feature = dataset.load_feature(video, method='c3d_fc7', l2=l2)
        audio_feature = dataset.load_feature(video, method='vgg', l2=l2)
        feature = np.hstack((rgb_feature, flow_feature, video_feature,audio_feature))
        X.append(feature)
        y.append(label)
        cnt+=1
        # if cnt>100:break
    X = np.array(X)
    y = np.array(y)
    # print(clf.predict(X),y)
    print(X.shape, y.shape)
    if l2:
        write_file = open(os.path.join(dataset.RES_DIR, "visual_audio_l2_svm.txt"), "w")
    else:
        write_file = open(os.path.join(dataset.RES_DIR, "visual_audio_svm.txt"), "w")
    for i in range(X.shape[0]):
        write_file.write('{}, {}, {}, {}\n'.format(y[i], 0, clf.predict(X[i].reshape(1, -1))[0], 0))
    write_file.close()

def evaluate():
    dataset = UCF101()
    #name+annotaion
    methods=[
        ("c3d_end","C3D End to END"),
        ("c3d_fc7_sgd", "C3D FC7 + SGD"),
        ("c3d_fc7_l2_sgd", "C3D FC7 + L2 + SGD"),
        ("c3d_fc7_svm","C3D FC7 + SVM"),
        ("c3d_fc7_l2_svm","C3D FC7 + L2 "),
        ("rgb_logit_svm","Inception "),
        ("rgb_logit_l2_svm","Inception + L2 "),
        ("flow_logit_svm","Flow "),
        ("flow_logit_l2_svm","Flow + L2 "),
        ("audio_svm","Audio"),
        ("audio_l2_svm", "Audio + L2 "),
        ("two_stream_svm","Two Stream "),
        ("two_stream_l2_svm","Two Stream + L2 "),
        ("visual_svm","Visual"),
        ("visual_l2_svm","Visual + L2 "),
        ("visual_audio_svm", "Visual + Audio"),
        ("visual_audio_l2_svm", "Visual + Audio + L2 ")
    ]
    result=[]
    for method,annotation in methods:
        with open(os.path.join(dataset.RES_DIR, method+'.txt'), 'r') as fr:
            total = 0
            cnt = 0
            for line in fr:
                label, _, pred, _ = line.strip().split()
                if label == pred:
                    cnt += 1
                total += 1
        result.append((annotation,cnt/total))
        print(annotation, "accuracy", cnt / total, "correct", cnt, "total", total)
    plot(result)

def plot(result,bar_width=0.5):
    feature_result = [
        ("C3D", 0.736),
        ("Inception ResNet", 0.758),
        ("Optical-Flow", 0.435),
        ("Audio", 0.234),
        ("Two-Stream", 0.777),
        ("Visual", 0.828),
        ("Multi-Modality", 0.872)
    ]
    classifier_result = [
        ("C3D End to End", 0.566),
        ("C3D FC7 + SGD", 0.716),
        ("C3D FC7 + SVM", 0.736)
    ]
    l2_result = [
        ("C3D", -0.009),
        ("Inception ResNet", -0.01),
        ("Optical-Flow", -0.104),
        ("Audio", -0.004),
        ("Two-Stream", 0.007),
        ("Visual", 0.028),
        ("Multi-Modality", 0.163)
    ]
    # plt.ylabel('Recall@AN=200', fontsize=fn_size)

    sns.set(font_scale=1.5,rc={'figure.figsize':(15,10)})
    sns.set_style('whitegrid')
    X=[x[1]*100 for x in feature_result ]
    y=[x[0] for x in feature_result]
    ax=sns.barplot(x=X,y=y,palette='PuBuGn_d')
    plt.xlim(0,100)
    for p in ax.patches:
        p.set_height(.5)
        ax.annotate("%.2lf " % p.get_width()+"%",xy=(p.get_x()+p.get_width()+1.2,p.get_y()+0.27))
    plt.xlabel("Accuracy")
    plt.savefig(os.path.join(UCF101.RES_DIR,'fig1'))
    plt.close()

    sns.set(font_scale=1.5,rc={'figure.figsize':(15,10)})
    sns.set_style('whitegrid')
    X=[x[1]*100 for x in l2_result ]
    y=[x[0] for x in l2_result]
    ax=sns.barplot(x=X,y=y,palette='PuBuGn_d')
    plt.xlim(-15,20)
    for p in ax.patches[:4]:
        p.set_height(.5)
        ax.annotate("%.2lf" % p.get_width()+"%", xy=(p.get_x() + p.get_width()-3.5, p.get_y() + 0.25))
    for p in ax.patches[4:]:
        p.set_height(.5)
        ax.annotate("%.2lf" % p.get_width()+"%", xy=(p.get_x() + p.get_width()+0.5, p.get_y() + 0.25))
    plt.xlabel("Accuracy")
    plt.savefig(os.path.join(UCF101.RES_DIR,'fig2'))
    plt.close()

    sns.set(font_scale=1.25,rc={'figure.figsize':(15,10)})
    sns.set_style('whitegrid')
    X=[x[1]*100 for x in classifier_result ]
    y=[x[0] for x in classifier_result]
    ax=sns.barplot(x=X,y=y,palette='PuBuGn_d')
    plt.xlim(0,100)
    for p in ax.patches:
        p.set_height(.5)
        ax.annotate("%.2lf" % p.get_width(), xy=(p.get_x() + p.get_width() + 1.2, p.get_y() + 0.25))
    plt.savefig(os.path.join(UCF101.RES_DIR,'fig3'))


def main(_):
    # fc7_sgd(l2=True)
    # fc7_svm()
    # end2end()
    # rgb_logit()
    # flow_logit(l2=False)
    # audio(l2=True)
    # two_stream(l2=True)
    # visual(l2=False)
    # visual_audio(l2=True)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
