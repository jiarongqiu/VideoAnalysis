#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-23

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import pickle
import json
import math


class ActivityNet(object):
    """"""
    FEATURE_DIR="/data1/densevid/ordered_feature/resnet200/raw"
    ANNOTATION_DIR = "/root/local_dir/annotation"
    MODEL_DIR="/root/local_dir/model"
    RES_DIR = "/root/local_dir/result"
    RESNET_FPS=8    # evert 8 frames
    def __init__(self,):
        """Constructor for ActivityNet"""
        # self.propocess()
        self.train_info=json.load(open(os.path.join(self.ANNOTATION_DIR, 'train.json'), 'r'))
        self.test_info=json.load(open(os.path.join(self.ANNOTATION_DIR, 'test.json'), 'r'))
        print("Train",len(self.train_info),'Test',len(self.test_info))

    def propocess(self):
        ANNOTATION_DIR = "/data1/densevid/captions"
        train_info,test_info={},{}
        info=json.load(open(os.path.join(ANNOTATION_DIR,'train.json'),'r'))
        for video,meta in info.iteritems():
            proposals=[]
            for start,end in meta['timestamps']:
                proposals.append({"start":start,'end':end})
            meta['proposals']=proposals
            meta['frame_duration']=meta['duration']/self.get_resnet_feature(video).shape[0]
            del meta['timestamps']
            del meta['sentences']
            train_info[video]=meta
        json.dump(train_info,open(os.path.join(self.ANNOTATION_DIR,'train.json'),'w'))
        info=json.load(open(os.path.join(ANNOTATION_DIR,'val_1.json'),'r'))
        for video,meta in info.iteritems():
            proposals=[]
            for start,end in meta['timestamps']:
                proposals.append({"start":start,'end':end})
            meta['proposals']=proposals
            meta['frame_duration'] = meta['duration'] / self.get_resnet_feature(video,train=False).shape[0]
            del meta['timestamps']
            del meta['sentences']
            test_info[video]=meta
        json.dump(test_info,open(os.path.join(self.ANNOTATION_DIR, 'test.json'), 'w'))
    
    def get_resnet_feature(self,video,train=True):
        if train:
            path=os.path.join(self.FEATURE_DIR,'trn',video+'.npy')
        else:
            path = os.path.join(self.FEATURE_DIR, 'val', video+'.npy')
        return np.load(open(path,'r'))

    def load_feature(self,video,start,end,train=True):
        if train:
            path=os.path.join(self.FEATURE_DIR,'trn',video+'.npy')
        else:
            path = os.path.join(self.FEATURE_DIR, 'val', video+'.npy')
        array=np.load(open(path,'r'))
        frame_duration = self.get_info(video,train=train)['frame_duration']
        start_pos=int(math.floor(start/frame_duration))
        end_pos=int(math.floor(end/frame_duration))
        feature=array[start_pos:end_pos,:]
        # norm = np.linalg.norm(feature, ord=2,axis=1)
        # feature /= norm.reshape(-1,1)
        return feature

    def get_info(self,video,train=True):
        if train:
            info=self.train_info[video]
        else:
            info=self.test_info[video]
        return info

    def get_train_test(self):
        return self.train_info,self.test_info
    def stat(self):
        ANNOTATION_DIR = "/data1/densevid/captions"
        train_info, test_info = {}, {}
        total_duration=cnt=0

        info = json.load(open(os.path.join(ANNOTATION_DIR, 'train.json'), 'r'))
        for video, meta in info.iteritems():
            total_duration+=meta['duration']
            cnt+=1

        print("Train", cnt)

        info = json.load(open(os.path.join(ANNOTATION_DIR, 'val_1.json'), 'r'))
        for video, meta in info.iteritems():
            total_duration+=meta['duration']
            cnt+=1
        print("Total", cnt)
        print("Total Duration", total_duration, "s", total_duration / 60.0 / 60.0,
              "h", "Average Duration", total_duration / cnt)


if __name__ == '__main__':
    dataset=ActivityNet()
    dataset.stat()
    # video="v_uh-H5Gmt4PI"
    # feature=dataset.get_resnet_feature(video=video)
    # print(feature.shape)
    # info=dataset.get_info(video)
    # print(info)
    # feature=dataset.load_feature(video,0,1)
    # print(feature.shape)