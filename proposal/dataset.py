
import numpy as np
from math import sqrt
import os
import random
import pickle

from data_provider.THUMOS14 import THUMOS14

import time

def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

class TrainingDataSet(object):
    def __init__(self,batch_size=128):
        self.dataset=THUMOS14()
        self.ctx_num=2
        self.unit_feature_size=4096
        self.unit_size=16.0
        self.unit_duration=1
        self.batch_size=batch_size
        self.visual_feature_dim=self.unit_feature_size*3
        self.feat_dir=self.dataset.FEATURE_DIR
        self.training_samples=self.prepare_training_sample()
        print("Training",len(self.training_samples))

    def prepare_training_sample(self):
        samples=[]
        train,_=self.dataset.load_in_info()
        for vid in train:
            used = set()
            info=train[vid]
            proposals=info['proposals']
            movie_name=vid
            for proposal in proposals:
                start=proposal['start']
                end=proposal['end']
                round_start=np.floor(start/self.unit_duration)*self.unit_duration
                round_end=np.floor(end/self.unit_duration)*self.unit_duration
                if round_start==round_end:
                    samples.append((movie_name, int(start), int(end), int(round_start), int(round_end), 1))
                    used.add(int(round_start))
                else:
                    while round_start<=round_end:
                        samples.append((movie_name, int(start), int(end), int(round_start), int(round_start+self.unit_duration), 1))
                        used.add(int(round_start))
                        round_start += self.unit_duration
            clip_start=0
            clip_end = clip_start + self.unit_duration
            while clip_end<info['duration']:
                if int(clip_start) not in used:
                    samples.append((movie_name, int(clip_start), int(clip_end), 0, 0, 0))
                clip_start += self.unit_duration
                clip_end = clip_start + self.unit_duration
        return sorted(samples)

    def next_batch(self):
        random_batch_index = random.sample(range(len(self.training_samples)), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        label_batch = np.zeros([self.batch_size], dtype=np.int32)
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32)
        index = 0
        begin=time.time()
        while index < self.batch_size:
            k = random_batch_index[index]
            movie_name = self.training_samples[k][0]
            if self.training_samples[k][5] == 1:
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                round_gt_start = self.training_samples[k][3]
                round_gt_end = self.training_samples[k][4]
                start_offset, end_offset = self.calculate_regoffset(clip_start, clip_end, round_gt_start, round_gt_end)
                feat = self.get_pooling_feature( movie_name, round_gt_start, round_gt_end)
                left_feat = self.get_left_context_feature( movie_name, round_gt_start, round_gt_end)
                right_feat = self.get_right_context_feature(movie_name, round_gt_start, round_gt_end)
                image_batch[index, :] = np.hstack((left_feat, feat, right_feat))
                duration = time.time() - begin
                label_batch[index] = 1
                offset_batch[index, 0] = start_offset
                offset_batch[index, 1] = end_offset
                index += 1
            else:
                clip_start = self.training_samples[k][1]
                clip_end = self.training_samples[k][2]
                left_feat = self.get_left_context_feature( movie_name, clip_start, clip_end)
                right_feat = self.get_right_context_feature( movie_name, clip_start, clip_end)
                feat = self.get_pooling_feature( movie_name, clip_start, clip_end)
                image_batch[index, :] = np.hstack((left_feat, feat, right_feat))
                duration = time.time() - begin
                label_batch[index] = 0
                offset_batch[index, 0] = 0
                offset_batch[index, 1] = 0
                index += 1
        return image_batch, label_batch, offset_batch

    def calculate_regoffset(self, clip_start, clip_end, round_gt_start, round_gt_end):
        start_offset = (round_gt_start - clip_start) / self.unit_duration
        end_offset = (round_gt_end - clip_end) / self.unit_duration
        return start_offset, end_offset

    '''
    Get the central features
    '''
    def get_pooling_feature(self,movie_name,start,end):
        swin_step=self.unit_duration
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        current_pos=start
        while current_pos<=end:
            swin_start=current_pos
            swin_end=current_pos+swin_step
            feat=self.dataset.load_feature(movie_name,swin_start,swin_end)
            all_feat=np.vstack((all_feat,feat))
            current_pos+=swin_step
        pool_feat=np.mean(all_feat,axis=0)
        return pool_feat

    '''
    Get the past (on the left of the central unit) context features
    '''

    def get_left_context_feature(self, movie_name, start, end):
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_start = start-self.unit_duration
        current_end=current_start+self.unit_duration
        while count < self.ctx_num:
            # print(movie_name,current_start,current_end)
            feat = self.dataset.load_feature(movie_name, current_start, current_end)
            all_feat = np.vstack((all_feat, feat))
            current_start -= self.unit_duration
            current_end = current_start + self.unit_duration
            count += 1
        pool_feat = np.mean(all_feat,axis=0)
        return pool_feat

    '''
    Get the future (on the right of the central unit) context features
    '''

    def get_right_context_feature(self, movie_name, start, end):
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_start = end
        current_end = current_start + self.unit_duration
        while count < self.ctx_num:
            feat = self.dataset.load_feature(movie_name, current_start, current_end)
            all_feat = np.vstack((all_feat, feat))
            current_start += self.unit_duration
            current_end = current_start + self.unit_duration
            count += 1
        pool_feat = np.mean(all_feat, axis=0)
        return pool_feat

class TestingDataSet(object):
    def __init__(self,batch_size=1):
        self.dataset=THUMOS14()
        #it_path: image_token_file path
        self.ctx_num=2
        self.unit_feature_size=4096
        self.unit_size=16.0
        self.unit_duration=1.0 #second
        self.batch_size=batch_size
        self.visual_feature_dim=self.unit_feature_size*3
        self.feat_dir=self.dataset.FEATURE_DIR
        self.samples=self.prepare_sample()
        self.idx=0
        print("Testing",len(self.samples))

    def prepare_sample(self):
        samples = []
        _, test = self.dataset.load_in_info()
        for vid in test:
            used = set()
            info = test[vid]
            proposals = info['proposals']
            movie_name = vid
            for proposal in proposals:
                start = proposal['start']
                end = proposal['end']
                round_start = np.floor(start / self.unit_duration) * self.unit_duration
                round_end = np.floor(end / self.unit_duration) * self.unit_duration
                # print(start,end,round_start,round_end)
                if round_start == round_end:
                    used.add(int(round_start))
                else:
                    while round_start <= round_end:
                        used.add(int(round_start))
                        round_start += self.unit_duration
            clip_start = 0
            clip_end = clip_start + self.unit_duration
            while clip_end < info['duration']:
                if int(clip_start) in used:
                    samples.append((movie_name, int(clip_start), int(clip_end), int(clip_start), int(clip_end),0))
                else:
                    samples.append((movie_name, 0,0,int(clip_start), int(clip_end), 0))
                clip_start += self.unit_duration
                clip_end = clip_start + self.unit_duration
        return sorted(samples)

    def get_sample(self,k):
        movie_name = self.samples[k][0]
        gt_start = self.samples[k][1]
        gt_end = self.samples[k][2]
        clip_start = self.samples[k][3]
        clip_end = self.samples[k][4]
        left_feat = self.get_left_context_feature( movie_name, clip_start, clip_end)
        right_feat = self.get_right_context_feature( movie_name, clip_start, clip_end)
        feat = self.get_pooling_feature( movie_name, clip_start, clip_end)
        image_batch = np.hstack((left_feat, feat, right_feat)).reshape(1,-1)
        # print(image_batch)
        return movie_name,gt_start,gt_end,clip_start,clip_end,image_batch

    def calculate_regoffset(self, clip_start, clip_end, round_gt_start, round_gt_end):
        start_offset = (round_gt_start - clip_start) / self.unit_duration
        end_offset = (round_gt_end - clip_end) / self.unit_duration
        return start_offset, end_offset

    '''
    Get the central features
    '''
    def get_pooling_feature(self,movie_name,start,end):
        swin_step=self.unit_duration
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        current_pos=start
        while current_pos<=end:
            swin_start=current_pos
            swin_end=current_pos+swin_step
            feat=self.dataset.load_feature(movie_name,swin_start,swin_end)
            all_feat=np.vstack((all_feat,feat))
            current_pos+=swin_step
        pool_feat=np.mean(all_feat,axis=0)
        return pool_feat

    '''
    Get the past (on the left of the central unit) context features
    '''

    def get_left_context_feature(self, movie_name, start, end):
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_start = start-self.unit_duration
        current_end=current_start+self.unit_duration
        while count < self.ctx_num:
            # print(movie_name,current_start,current_end)
            feat = self.dataset.load_feature(movie_name, current_start, current_end)
            all_feat = np.vstack((all_feat, feat))
            current_start -= self.unit_duration
            current_end = current_start + self.unit_duration
            count += 1
        pool_feat = np.mean(all_feat,axis=0)
        return pool_feat

    '''
    Get the future (on the right of the central unit) context features
    '''

    def get_right_context_feature(self, movie_name, start, end):
        all_feat = np.zeros([0, self.unit_feature_size], dtype=np.float32)
        count = 0
        current_start = end
        current_end = current_start + self.unit_duration
        while count < self.ctx_num:
            feat = self.dataset.load_feature(movie_name, current_start, current_end)
            all_feat = np.vstack((all_feat, feat))
            current_start += self.unit_duration
            current_end = current_start + self.unit_duration
            count += 1
        pool_feat = np.mean(all_feat, axis=0)
        return pool_feat
if __name__ == '__main__':
    dataset=TrainingDataSet()
    image_batch,label_batch,offset_batch=dataset.next_batch()
    # print(image_batch.shape,image_batch[0,:4096],image_batch[0,4097:8172],image_batch[0,8173:])
    # print(label_batch.shape,label_batch)
    print(offset_batch.shape,offset_batch)
    # dataset=TestingDataSet()
    # print(dataset.get_sample(0))



