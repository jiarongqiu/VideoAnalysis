#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 08/03/2018

from __future__ import print_function
from __future__ import division


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from data_provider.UCF11 import UCF11
from data_provider.utils import get_file_list
from features import *
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate

DIR_NAME="../data/"
FOLDER="feature_evaluation/"

RGB_EMBEDDING_TRAIN_X=DIR_NAME+FOLDER+'ucf11.rgb.train.x'
RGB_EMBEDDING_TRAIN_Y=DIR_NAME+FOLDER+'ucf11.rgb.train.y'
RGB_EMBEDDING_TEST_X=DIR_NAME+FOLDER+'ucf11.rgb.test.x'
RGB_EMBEDDING_TEST_Y=DIR_NAME+FOLDER+'ucf11.rgb.test.y'
FLOW_EMBEDDING_TRAIN_X=DIR_NAME+FOLDER+'ucf11.flow.train.x'
FLOW_EMBEDDING_TRAIN_Y=DIR_NAME+FOLDER+'ucf11.flow.train.y'
FLOW_EMBEDDING_TEST_X=DIR_NAME+FOLDER+'ucf11.flow.test.x'
FLOW_EMBEDDING_TEST_Y=DIR_NAME+FOLDER+'ucf11.flow.test.y'
AUDIO_EMBEDDING_TRAIN_X=DIR_NAME+FOLDER+'ucf11.audio.train.x'
AUDIO_EMBEDDING_TRAIN_Y=DIR_NAME+FOLDER+'ucf11.audio.train.y'
AUDIO_EMBEDDING_TEST_X=DIR_NAME+FOLDER+'ucf11.audio.test.x'
AUDIO_EMBEDDING_TEST_Y=DIR_NAME+FOLDER+'ucf11.audio.test.y'

def RGB_mean_process(train_cnt=100,test_cnt=10):
    rgb=RGBFeature()
    dataset = UCF11()
    clips, y = dataset.get_data(split_name='train')
    X = []
    for i in range(train_cnt):
        print(round(i / train_cnt, 3))
        clip=clips[i]
        X.append(np.mean(rgb.images2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:train_cnt])
    X.dump(RGB_EMBEDDING_TRAIN_X)
    y.dump(RGB_EMBEDDING_TRAIN_Y)

    clips, y = dataset.get_data(split_name='test')
    X = []
    for i in range(test_cnt):
        print(round(i / test_cnt, 3))
        clip = clips[i]
        X.append(np.mean(rgb.images2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:test_cnt])
    X.dump(RGB_EMBEDDING_TEST_X)
    y.dump(RGB_EMBEDDING_TEST_Y)

def flow_mean_process(train_cnt=100, test_cnt=10):
    flow = FlowFeature()
    dataset = UCF11()
    clips, y = dataset.get_data(split_name='train')
    X = []
    for i in range(train_cnt):
        print(round(i / train_cnt, 3))
        clip = clips[i]
        X.append(np.mean(flow.images2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:train_cnt])
    X.dump(FLOW_EMBEDDING_TRAIN_X)
    y.dump(FLOW_EMBEDDING_TRAIN_Y)

    clips, y = dataset.get_data(split_name='test')
    X = []
    for i in range(test_cnt):
        print(round(i / test_cnt, 3))
        clip = clips[i]
        X.append(np.mean(flow.images2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:test_cnt])
    X.dump(FLOW_EMBEDDING_TEST_X)
    y.dump(FLOW_EMBEDDING_TEST_Y)

def audio_mean_process(train_cnt=100, test_cnt=10):
    audio = AudioFeature()
    dataset = UCF11()
    clips, y = dataset.get_data(split_name='train')
    X = []
    for i in range(train_cnt):
        print(round(i / train_cnt, 3))
        clip = clips[i]
        X.append(np.mean(audio.wav2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:train_cnt])
    X.dump(AUDIO_EMBEDDING_TRAIN_X)
    y.dump(AUDIO_EMBEDDING_TRAIN_Y)

    clips, y = dataset.get_data(split_name='test')
    X = []
    for i in range(test_cnt):
        print(round(i / test_cnt, 3))
        clip = clips[i]
        X.append(np.mean(audio.images2feature(get_file_list(clip)), axis=0))
    X = np.array(X)
    y = np.array(y[:test_cnt])
    X.dump(AUDIO_EMBEDDING_TEST_X)
    y.dump(AUDIO_EMBEDDING_TEST_Y)

def RGBSVC():
    train_X, train_y, test_X, test_y=np.load(RGB_EMBEDDING_TRAIN_X),np.load(RGB_EMBEDDING_TRAIN_Y),np.load(RGB_EMBEDDING_TEST_X),np.load(RGB_EMBEDDING_TEST_Y)
    print(train_X.shape, train_y.shape)
    # clf = SVC()
    # clf.fit(train_X, train_y)
    # Y = clf.predict(test_X)
    # print(accuracy_score(test_y, Y))

def flowSVC():
    train_X, train_y, test_X, test_y = np.load(FLOW_EMBEDDING_TRAIN_X), np.load(FLOW_EMBEDDING_TRAIN_Y), np.load(
        FLOW_EMBEDDING_TEST_X), np.load(FLOW_EMBEDDING_TEST_Y)

    print(train_X.shape, train_y.shape)
    # clf = SVC()
    # clf.fit(train_X, train_y)
    # Y = clf.predict(test_X)
    # print(accuracy_score(test_y, Y))

def main():
    # RGB_mean_process()
    # flow_mean_process()
    RGBSVC()
    # flowSVC()



if __name__ == '__main__':
    main()
