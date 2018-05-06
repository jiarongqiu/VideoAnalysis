#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-3

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random

from activity_net.ActivityNet import ActivityNet
from activity_net.evaluate import cal_average_recall
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import pickle
import os

def kmeans_cluster(data,clusters_num=10,top_k=10):
    kmeans=KMeans(n_clusters=clusters_num,random_state=0,n_init=5).fit(data.reshape(-1,1))
    idx=[idx for idx, _ in Counter(kmeans.labels_).most_common(top_k)]
    # print(Counter(kmeans.labels_).most_common(top_k))
    return np.array(kmeans.cluster_centers_)[idx].reshape(-1)

# def random_proposal(train, test, avg_num_proposals=100):
#     data=[]
#     for vid, info in train.iteritems():
#         for proposal in info['proposals']:
#             data.append(proposal['end']-proposal['start'])
#     data=np.array(data)
#     windows=kmeans_cluster(data)
#     ret={}
#     for vid,info in test.iteritems():
#         proposal=[]
#         for _ in range(avg_num_proposals):
#             center=info['duration']*random.random()
#             length=random.choice(windows)
#             start=max(center-length/2,0)
#             end=min(center+length/2,info['duration'])
#             proposal.append({'start':start,'end':end})
#         random.shuffle(proposal)
#         ret[vid]={"proposals":proposal}
#     pickle.dump(ret, open(os.path.join(THUMOS14.RES_DIR, 'random_proposal'), 'w'))
#     print(cal_average_recall(ret, test, num_proposals=200))
#     return ret

def sliding_window_kmeans(train,val,stride_ratio=0.5):
    data=[]
    for vid, info in train.iteritems():
        for proposal in info['proposals']:
            data.append(proposal['end']-proposal['start'])
    data=np.array(data)
    windows=kmeans_cluster(data,clusters_num=20,top_k=20)
    # print(windows)
    # for idx,(vid,info) in enumerate(train.iteritems()):
    #     if idx>10:break
    #     frame_duration=info['frame_duration']
    #     print(frame_duration,frame_duration/16*2048)
    ret={}
    for vid,info in val.iteritems():
        proposal=[]
        duration=info['duration']
        for window in windows:
            start = 0
            end=window
            while start<end and start<duration and end <duration:
                proposal.append({'start': start, 'end': end,'score':0})
                start+=window*stride_ratio
                end+=window*stride_ratio
        random.shuffle(proposal)
        ret[vid]={"proposals":proposal}
    # pickle.dump(ret, open(os.path.join(THUMOS14.RES_DIR, 'sliding_window_proposal'), 'w'))
    # print(cal_average_recall(ret, test, num_proposals=200))
    return ret

# def even_segment(train, test):
#     ret = {}
#     for vid, info in test.iteritems():
#         proposals = []
#         duration = info['duration']
#         windows=[duration/i for i in range(1,20+1)]
#         for window in windows:
#             start = 0
#             end = window
#             while start < end and start < duration:
#                 proposals.append({'start': start, 'end': end})
#                 start += window
#                 end = min(end+window,duration)
#         random.shuffle(proposals)
#         ret[vid] = {"proposals": proposals}
#     pickle.dump(ret, open(os.path.join(THUMOS14.RES_DIR, 'even_proposal'), 'w'))
#     print(cal_average_recall(ret, test, num_proposals=200))
#     return ret


if __name__ == '__main__':
    # data=THUMOS14()
    dataset=ActivityNet()
    train,val=dataset.train_info,dataset.val_info
    # train,test=data.load_in_info()
    # random_proposal(train,test)
    ret=sliding_window_kmeans(train,val)
    # even_segment(train,test)
    print(cal_average_recall(ret,val,num_proposals=100))