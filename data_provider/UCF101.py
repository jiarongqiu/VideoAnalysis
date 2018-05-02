#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-6

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_provider.dataset import Dataset
from data_provider.utils import *
from video.flow_test import dense_optical_flow_test
from audio.audio_provider import VGG
import random


class UCF101(Dataset):
    """"""

    DIR_NAME = '/home/charlie/Desktop/dataset/UCF-101/'
    LABEL_LIST = '/home/charlie/Desktop/dataset/UCF-101/annotation/label.list'
    TRAIN_LIST = '/home/charlie/Desktop/dataset/UCF-101/annotation/train.list'
    TEST_LIST = '/home/charlie/Desktop/dataset/UCF-101/annotation/test.list'
    VIDEO_DIR = '/home/charlie/Desktop/dataset/UCF-101/video/'
    IMAGE_DIR = '/home/charlie/Desktop/dataset/UCF-101/image/'
    RES_DIR = '/home/charlie/Desktop/dataset/UCF-101/result/'
    FEATURE_DIR='/home/charlie/Desktop/dataset/UCF-101/feature/'
    FLOW_DIR='/home/charlie/Desktop/dataset/UCF-101/flow/'
    AUDIO_DIR='/home/charlie/Desktop/dataset/UCF-101/audio/'
    NUM_CLASSES = 101
    BATCH_SIZE = 16
    NUM_PER_CLIP = 16
    FPS = 4
    CROP_SIZE = 299

    def __init__(self, ):
        """Constructor for UCF101"""
    def preprocess(self):
        video2label = {}
        train = []
        test = []
        with open(os.path.join(self.DIR_NAME, 'annotation', 'C3D', 'train.list'), 'r') as fr:
            for line in fr:
                path, id = line.strip().split()
                id = int(id)
                video = path.split('/')[-1]
                name = video.split('_')[1]
                # print(name,video)
                video2label[name] = id
                train.append((video, id))
        with open(os.path.join(self.DIR_NAME, 'annotation', 'C3D', 'test.list'), 'r') as fr:
            for line in fr:
                path, id = line.strip().split()
                id = int(id)
                video = path.split('/')[-1]
                name = video.split('_')[1]
                # print(name,video)
                video2label[name] = id
                test.append((video, id))
        print(len(video2label))

        fw = open(self.LABEL_LIST, 'w')
        for video,idx in sorted(list(video2label.iteritems()), key=lambda x: x[1]):
            fw.write(video + " " + str(idx) + "\n")
        fw.close()

        print("Train",len(train),"Test",len(test))

        #output train/test_list
        fw=open(self.TRAIN_LIST,'w')
        for video,label in train:
            name=video.split('_')[1]
            path=os.path.join(self.VIDEO_DIR,name,video+'.avi')
            out = os.path.join(self.IMAGE_DIR, video)
            wav_out = os.path.join(self.AUDIO_DIR, video + '.wav')
            single_video_to_wav(path,wav_out)
            # single_video_to_image(path, out, frame=self.FPS, crop_size=self.CROP_SIZE)
            fw.write(out+" "+str(label)+'\n')
        fw.close()

        fw=open(self.TEST_LIST,'w')
        for video,label in test:
            name = video.split('_')[1]
            path = os.path.join(self.VIDEO_DIR,name,video+'.avi')
            out = os.path.join(self.IMAGE_DIR, video)
            wav_out=os.path.join(self.AUDIO_DIR,video+'.wav')
            # single_video_to_image(path, out, frame=self.FPS, crop_size=self.CROP_SIZE)
            single_video_to_wav(path,wav_out)
            fw.write(out+" "+str(label)+'\n')
        fw.close()

    def load_in(self):
        train_list = []
        with open(self.TRAIN_LIST, 'r') as fr:
            for line in fr:
                video, label = line.strip().split()
                train_list.append((video, int(label)))
        test_list = []
        with open(self.TEST_LIST, 'r') as fr:
            for line in fr:
                video, label = line.strip().split()
                test_list.append((video, int(label)))
        self.train_list = train_list
        self.test_list = test_list

    def get_next_batch(self, batch_size=None):
        if not batch_size: batch_size = self.BATCH_SIZE

    def get_random_clip(self, folder):
        frames = sorted(get_file_list(folder))
        if len(frames) < 16:
            while len(frames) < 16:
                frames.append(frames[-1])
            return frames
        else:
            start = random.randint(0, len(frames) - self.NUM_PER_CLIP)
            return frames[start:start + self.NUM_PER_CLIP]

    def load_feature(self,video,method,l2=False):
        path=os.path.join(self.FEATURE_DIR,method,video+'.np')
        feature=np.load(path)
        if l2:
            norm=np.linalg.norm(feature,ord=2)
            if norm!=0:feature/=norm
        return feature

    def get_frames(self,folder):
        frames = sorted(get_file_list(folder))
        return frames

    def flow_process(self):
        self.load_in()
        for video, label in self.train_list:
            name=video.split('/')[-1]
            out=os.path.join(self.FLOW_DIR,name)
            dense_optical_flow_test(self.get_frames(video),out)
        for video, label in self.test_list:
            name=video.split('/')[-1]
            out=os.path.join(self.FLOW_DIR,name)
            dense_optical_flow_test(self.get_frames(video),out)

    def audio_process(self):
        provider=VGG()
        self.load_in()
        for video, label in self.train_list:
            name=video.split('/')[-1]
            audio_path=os.path.join(self.AUDIO_DIR,name+'.wav')
            out=os.path.join(self.FEATURE_DIR,'vgg',name+'.np')
            feat=provider.wav2feature(audio_path)
            feat.dump(open(out,'w'))
        for video, label in self.test_list:
            name = video.split('/')[-1]
            audio_path = os.path.join(self.AUDIO_DIR, name + '.wav')
            out = os.path.join(self.FEATURE_DIR, 'vgg', name + '.np')
            feat = provider.wav2feature(audio_path)
            feat.dump(open(out, 'w'))

    def stat(self):
        self.load_in()
        total_duration=0
        cnt=0
        for video, _ in self.train_list:
            name = video.split('/')[-1]
            folder=name.split('_')[1]
            file=os.path.join(self.VIDEO_DIR,folder,name+'.avi')
            total_duration+=get_video_duration(file)
            cnt+=1
        for video, label in self.test_list:
            name = video.split('/')[-1]
            folder = name.split('_')[1]
            file = os.path.join(self.VIDEO_DIR, folder, name + '.avi')
            total_duration += get_video_duration(file)
            cnt += 1
        print("Total Duration",total_duration,"Average Duration",total_duration/cnt)



if __name__ == '__main__':
    dataset = UCF101()
    dataset.stat()
    # dataset.audio_process()
    # dataset.preprocess()
    # dataset.flow_process()
    # dataset.c3d_fc6_feature()
    # print(dataset.load_feature("v_ApplyEyeMakeup_g01_c01",'rgb_logit',l2=True))
    # print(dataset.load_feature("v_ApplyEyeMakeup_g01_c01", 'flow_logit', l2=True))
    # print(dataset.load_feature("v_Archery_g05_c01", 'rgb_logit', l2=True))
    # dataset.get_flow_frames("/home/charlie/Desktop/dataset/UCF-101/image/v_Archery_g09_c03")
