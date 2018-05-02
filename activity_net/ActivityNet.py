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

# Implementation of Activity Net Task 1
#

class ActivityNet(object):
    """"""
    ANNOTATION="/data3/densevid/proposals/activity_net.v1-3.min.json"
    FEATURE_DIR="/data3/densevid/ordered_feature/resnet200"
    VIDEO_DIR="/data3/densevid/videos/"
    ANNOTATION_DIR = "/root/charlie_remote/annotation"
    MODEL_DIR="/root/charlie_remote/model"
    RES_DIR = "/root/charlie_remote/result"

    def __init__(self,):
        """Constructor for ActivityNet"""
        # self.preprocess()
        self.train_info=json.load(open(os.path.join(self.ANNOTATION_DIR, 'train.json'), 'r'))
        self.test_info=json.load(open(os.path.join(self.ANNOTATION_DIR, 'test.json'), 'r'))
        self.val_info=json.load(open(os.path.join(self.ANNOTATION_DIR, 'val.json'), 'r'))
        print("Train",len(self.train_info),'Test',len(self.test_info),'Val',len(self.val_info))

    def preprocess(self):
        def get_post(self, vid, subset):
            name = "v_" + vid + ".mp4"
            if subset == 'training':
                path = os.path.join(self.VIDEO_DIR, "trn", name)
            elif subset == 'testing':
                path = os.path.join(self.VIDEO_DIR, "tst", name)
            elif subset == 'validation':
                path = os.path.join(self.VIDEO_DIR, "val", name)
            else:
                print("ERROR", subset)
                path = ""
            if os.path.exists(path): return name
            return "v_" + vid + ".webm"
        info=json.load(open(self.ANNOTATION,'r'))['database']
        train,val,test={},{},{}
        for video,meta in info.iteritems():
            tmp = {}
            subset=meta['subset']
            video=get_post(video,subset)
            proposals=[]
            for proposal in meta['annotations']:
                start,end=proposal['segment']
                proposals.append({"start":start,'end':end})
            tmp['proposals']=proposals
            tmp['frame_duration'] = meta['duration'] / self.get_resnet_feature(video,subset=subset).shape[0]
            tmp['duration']=meta['duration']
            if subset=='training':
                train[video]=tmp
            elif subset=='testing':
                test[video]=tmp
            elif subset=='validation':
                val[video]=tmp
            else:
                print("ERROR",subset)
        print("Train",len(train),"Val",len(val),'Test',len(test))
        json.dump(train,open(os.path.join(self.ANNOTATION_DIR,'train.json'),'w'))
        json.dump(val, open(os.path.join(self.ANNOTATION_DIR, 'val.json'), 'w'))
        json.dump(test,open(os.path.join(self.ANNOTATION_DIR, 'test.json'), 'w'))

    def get_resnet_feature(self,video,subset="training"):
        if subset == 'training':
            path = os.path.join(self.FEATURE_DIR, 'trn', video + '.npy')
        elif subset == 'testing':
            path = os.path.join(self.FEATURE_DIR, 'tst', video + '.npy')
        elif subset == 'validation':
            path = os.path.join(self.FEATURE_DIR, 'val', video + '.npy')
        else:
            print("ERROR", subset)
            path=""
        return np.load(open(path,'r'))

    def load_feature(self,video,start,end,subset):
        array=self.get_resnet_feature(video,subset)
        frame_duration = self.get_info(video,subset)['frame_duration']
        start_pos=int(math.floor(start/frame_duration))
        end_pos=int(math.floor(end/frame_duration))
        feature=array[start_pos:end_pos,:]
        # norm = np.linalg.norm(feature, ord=2,axis=1)
        # feature /= norm.reshape(-1,1)
        return feature

    def get_info(self,video,subset):
        if subset == 'training':
            info = self.train_info[video]
        elif subset == 'testing':
            info = self.test_info[video]
        elif subset == 'validation':
            info = self.val_info[video]
        else:
            info={}
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
    # dataset.stat()
    video="v_2tO1ApNwXpQ.mp4"
    subset='validation'
    feature=dataset.get_resnet_feature(video=video,subset=subset)
    print(feature.shape)
    info=dataset.get_info(video,subset)
    print(info)
    feature=dataset.load_feature(video,0,1,subset)
    print(feature[0,:])