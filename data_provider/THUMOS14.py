#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-3

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import json
import numpy as np

from data_provider.dataset import Dataset
from data_provider.utils import *
from video.video_provider import C3D

class THUMOS14(Dataset):
    """"""
    DIR = "/home/charlie/Desktop/dataset/THUMOS-14/"
    LABEL_LIST = '/home/charlie/Desktop/dataset/THUMOS-14/annotation/label.list'
    TRAIN_LIST = '/home/charlie/Desktop/dataset/THUMOS-14/annotation/train.list'
    TEST_LIST = '/home/charlie/Desktop/dataset/THUMOS-14/annotation/test.list'
    VIDEO_DIR= '/home/charlie/Desktop/dataset/THUMOS-14/video/'
    IMAGE_DIR = '/home/charlie/Desktop/dataset/THUMOS-14/image/'
    RES_DIR='/home/charlie/Desktop/dataset/THUMOS-14/result/'
    FEATURE_DIR="/home/charlie/Desktop/dataset/THUMOS-14/feature/"
    NUM_CLASSES = 101
    BATCH_SIZE = 16
    FEATURE_DIM=4096
    FPS=16
    CROP_SIZE=299

    def __init__(self, ):
        """Constructor for THUMOS14"""

    def create_train_test_info(self):
        """{
            "video_id"{
            "path"
            "duration":
            "proposals":[
                {'start': , 'end': },
                ....
                ]
            }}"""
        ret_json = {}
        annotation_dir = os.path.join(self.DIR, 'annotation', 'validation')
        annotation_files = get_file_list(annotation_dir)
        for file in annotation_files:
            with open(file, 'r') as fr:
                label = self.get_label_from_path(file)
                for line in fr:
                    line = line.strip().split()
                    video_id, start, end = line[0], float(line[1]), float(line[2])
                    assert start < end
                    info = ret_json.get(video_id, {})
                    proposals = info.get('proposals', [])
                    proposals.append({'start': start, 'end': end, 'label': label})
                    info['proposals'] = proposals
                    ret_json[video_id] = info
        dir = os.path.join(self.VIDEO_DIR, 'validation')
        for file in get_file_list(dir):
            video_id = file.split('/')[-1].split('.')[0]
            if video_id not in ret_json: continue
            info = ret_json[video_id]
            info["duration"] = get_video_duration(file)
            info["path"] = file
            ret_json[video_id] = info
        json.dump(ret_json, open(os.path.join(self.DIR, 'annotation', 'train_json'), 'w'))
        ret_json = {}
        annotation_dir = os.path.join(self.DIR, 'annotation', 'test')
        annotation_files = get_file_list(annotation_dir)
        for file in annotation_files:
            with open(file, 'r') as fr:
                label = self.get_label_from_path(file)
                for line in fr:
                    line = line.strip().split()
                    video_id, start, end = line[0], float(line[1]), float(line[2])
                    assert start < end
                    info = ret_json.get(video_id, {})
                    proposals = info.get('proposals', [])
                    proposals.append({'start': start, 'end': end, 'label': label})
                    info['proposals'] = proposals
                    ret_json[video_id] = info
        dir = os.path.join(self.VIDEO_DIR, 'test')
        for file in get_file_list(dir):
            video_id = file.split('/')[-1].split('.')[0]
            if video_id not in ret_json: continue
            info = ret_json[video_id]
            info["duration"] = get_video_duration(file)
            info["path"] = file
            ret_json[video_id] = info
        json.dump(ret_json, open(os.path.join(self.DIR, 'annotation', 'test_json'), 'w'))

    def load_in_info(self):
        return json.load(open(os.path.join(self.DIR, 'annotation', 'train_json'), 'r')), json.load(
            open(os.path.join(self.DIR, 'annotation', 'test_json'), 'r'))
    def load_in_label(self):
        label2idx={}
        with open(self.LABEL_LIST,'r') as fr:
            for line in fr:
                idx,label=line.strip().split()
                idx=int(idx)
                label2idx[label]=idx
        return label2idx
    def preprocess(self):
        train,test=self.load_in_info()
        for vid in train:
            info=train[vid]
            path=info['path']
            single_video_to_image(path,os.path.join(self.IMAGE_DIR,vid),frame=self.FPS)
        for vid in test:
            info=test[vid]
            path=info['path']
            single_video_to_image(path,os.path.join(self.IMAGE_DIR,vid),frame=self.FPS)

    def c3d_fc7(self, unit_size=16):
        '''
        extract video feature every 16 frames
        :return: feature:vid_start_end.np, where start(ms) and end(ms) ...
        '''
        c3d=C3D()
        train, test = self.load_in_info()
        # for idx,vid in enumerate(train):
        #     print("Process",idx/len(train)*100)
        #     info=train[vid]
        #     path=info['path']
        #     folder=os.path.join(self.FEATURE_DIR,'c3d_fc7',vid)
        #     create_folder(folder)
        #     frames=get_file_list(os.path.join(self.IMAGE_DIR,vid),sort=True)
        #     span=1/self.FPS*60
        #     start = 0
        #     end = unit_size
        #     while end<len(frames):
        #         start_time=int(start*span)
        #         end_time=int(end*span)
        #         feature=c3d.get_feature(frames[start:end])
        #         naming=os.path.join(folder,"_".join([vid,str(start_time),str(end_time)])+".np")
        #         feature.dump(naming)
        #         start+=unit_size
        #         end+=unit_size
        for idx,vid in enumerate(test):
            print("Process",idx/len(test)*100)
            folder=os.path.join(self.FEATURE_DIR,'c3d_fc7',vid)
            create_folder(folder)
            frames=get_file_list(os.path.join(self.IMAGE_DIR,vid),sort=True)
            span=1/self.FPS*60
            start = 0
            end = unit_size
            while end<len(frames):
                start_time=int(start*span)
                end_time=int(end*span)
                feature=c3d.get_feature(frames[start:end])
                naming=os.path.join(folder,"_".join([vid,str(start_time),str(end_time)])+".np")
                feature.dump(naming)
                start+=unit_size
                end+=unit_size


    def load_feature(self,vid,start,end,method='c3d_fc7',l2=False):
        '''
        :param vid:
        :param start: s
        :param end: s
        :return:
        '''
        start=int(start*60)
        end=int(end*60)
        naming=os.path.join(self.FEATURE_DIR,method,vid,"_".join([vid,str(start),str(end)])+'.np')
        if os.path.exists(naming):
            feature=np.load(naming)
            if l2:
                norm = np.linalg.norm(feature, ord=2)
                feature /= norm
            return feature
        else:
            return np.zeros([self.FEATURE_DIM], dtype=np.float32)

    def get_label_from_path(self, path):
        return os.path.splitext(path)[0].split('/')[-1].split('_')[0].lower()

    def create_label_list(self):
        labels = [self.get_label_from_path(label) for label in
                  get_file_list(os.path.join(self.DIR, "annotation", "validation"))]
        with open(self.LABEL_LIST, 'w') as fw:
            for idx, label in enumerate(labels):
                fw.write(str(idx) + " " + label + "\n")

    def create_finetune_info(self):
        label2idx={}
        with open(self.LABEL_LIST,'r') as fr:
            for line in fr:
                idx,label=line.strip().split()
                idx=int(idx)
                label2idx[label]=idx
        # print(label2idx)
        train,test=self.load_in_info()
        train_list,test_list=[],[]
        for vid in train:
            proposals=train[vid]['proposals']
            for p in proposals:
                start=p['start']
                end=p['end']
                label=p['label']
                idx=label2idx[label]
                train_list.append([vid,start,end,idx])

        for vid in test:
            proposals = test[vid]['proposals']
            for p in proposals:
                start = p['start']
                end = p['end']
                label = p['label']
                idx = label2idx[label]
                test_list.append([vid,start, end, idx])
        json.dump(train_list, open(self.TRAIN_LIST, 'w'))
        json.dump(test_list, open(self.TEST_LIST, 'w'))

    def load_finetune_info(self):
        return json.load(open(self.TRAIN_LIST, 'r')),json.load(open(self.TEST_LIST, 'r'))

    def get_frames(self,vid,start,end):
        start = start
        end = end
        frames=[]
        for file in sorted(get_file_list(os.path.join(self.IMAGE_DIR,vid))):
            time=int(file.split('/')[-1].split('.')[0])*1.0/self.FPS
            if time>start and time<end:frames.append(file)
        if len(frames)==0: print(vid,start, end)
        return frames

    def stat(self):
        train, test = self.load_in_info()
        total_duration=0
        cnt=0
        for vid,info in train.iteritems():
            total_duration+=info['duration']
            cnt+=1
        print("Train",cnt)
        for vid,info in test.iteritems():
            total_duration+=info['duration']
            cnt+=1
        print("Total",cnt)
        print("Total Duration",total_duration,"s",total_duration/60.0/60.0,
              "h","Average Duration",total_duration/cnt)

if __name__ == '__main__':
    dataset = THUMOS14()
    dataset.stat()
    # dataset.create_label_list()
    # dataset.create_train_test_info()
    # dataset.preprocess()
    # dataset.c3d_fc7()
    # print(dataset.load_feature("video_test_0000004",0,1.0)[:50])
    # dataset.create_finetune_info()
    # print(len(dataset.get_frames("video_validation_0000856",85.9,91.5)))