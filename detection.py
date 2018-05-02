#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-24

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from data_provider.THUMOS14 import THUMOS14
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import numpy as np
import pickle
import os
from collections import Counter
import pandas as pd
from proposal.evaluate import cal_iou
import time

def mAP(ground_truth, proposals, num_proposals=200, tiou=0.5):
    '''
    mean average precision of each category
    :param groud_truth: video,start,end,label
    :param proposals:video,start,end,label
    :return:
    '''
    proposals = pd.DataFrame(proposals)
    proposals = proposals.sort_values(4, ascending=False)
    ground_truth = pd.DataFrame(ground_truth)
    videos = pd.Series.unique(ground_truth.loc[:, 0])
    if len(videos) * num_proposals < len(proposals):
        proposals = proposals.loc[:len(videos) * num_proposals, :]
    print(len(proposals))
    actions = sorted(pd.Series.unique(ground_truth.loc[:, 3]))
    precision=[]
    for action in actions:
        y_pred = proposals.loc[proposals.loc[:, 3] == action]
        y = ground_truth.loc[ground_truth.loc[:, 3] == action]
        y_pred = y_pred.sort_values(4, ascending=False)
        precision_recall = []
        right = fetch = total=0
        cnt=0
        used=set()
        for idx, row in y_pred.iterrows():
            total+=1
            video, start, end, label, _ = row
            cnt+=1
            # if cnt>50:break
            for idx, row in y.loc[y.loc[:, 0] == video].iterrows():
                if idx in used:continue
                _, ground_truth_start, ground_truth_end, ground_truth_label = row
                # print(start,end,ground_truth_start,ground_truth_end,cal_iou((start, end), (ground_truth_start, ground_truth_end)))
                if cal_iou((start, end), (ground_truth_start, ground_truth_end)) > tiou:
                    # print(idx,ground_truth_label,label,cal_iou((start, end), (ground_truth_start, ground_truth_end)))
                    fetch += 1
                    used.add(idx)
                    if ground_truth_label == label:
                        right += 1
                    break
            if fetch == 0:
                precision_recall.append((0, 0))
            else:
                precision_recall.append((right / total, fetch))
        precision_recall=pd.DataFrame(precision_recall)
        grouped=precision_recall.loc[:,0].groupby(precision_recall.loc[:,1])
        grouped_precision=grouped.max()
        precision.append(sum(grouped_precision)/len(grouped_precision))
        # break
    print(sum(precision)/len(precision),precision)



def train():
    dataset = THUMOS14()
    train, test = dataset.load_in_info()
    label2idx = dataset.load_in_label()
    X, y = [], []
    cnt = 0
    for video, info in train.iteritems():
        proposals = info['proposals']
        for p in proposals:
            start = math.floor(p['start'])
            end = math.floor(p['end'])
            label = p['label']
            idx = label2idx[label]
            while start <= end:
                feature = dataset.load_feature(video, start, start + 1)
                X.append(feature)
                y.append(idx)
                start += 1
        # cnt += 1
        # if cnt > 10: break
    X = np.array(X)
    y = np.array(y)
    clf = SVC(kernel='linear')
    clf.fit(X=X, y=y)
    pred = clf.predict(X)
    print(X.shape, y.shape)
    print(accuracy_score(y_true=y, y_pred=pred))
    # pickle.dump(clf, open(os.path.join(THUMOS14.MODEL_DIR, 'detection_svm'), 'w'))


def test():
    clf = pickle.load(open(os.path.join(THUMOS14.MODEL_DIR, 'detection_svm'), 'r'))
    dataset = THUMOS14()
    _, test = dataset.load_in_info()
    label2idx = dataset.load_in_label()

    ground_truth, proposals = [], []
    result = pickle.load(open(os.path.join(THUMOS14.RES_DIR, 'turn_proposal'), 'r'))
    for video, info in result.iteritems():
        candidates = info['proposals']
        print(video, len(candidates))
        for proposal in candidates:
            start = math.floor(proposal['start'])
            end = math.floor(proposal['end'])
            score = proposal['score']
            if start>end:continue
            pt=start
            X = []
            while pt <= end:
                feature = dataset.load_feature(video, pt, pt + 1)
                X.append(feature)
                pt += 1
            X = np.array(X)
            pred = clf.predict(X)
            most_common, _ = Counter(pred).most_common(1)[0]
            proposal['label'] = most_common
            proposals.append((video, start, end, most_common, score))
    for video, info in test.iteritems():
        for p in info['proposals']:
            start = p['start']
            end = p['end']
            label = p['label']
            idx = label2idx[label]
            ground_truth.append((video, start, end, idx))
    pickle.dump([ground_truth, proposals], open(os.path.join(THUMOS14.RES_DIR, 'detection_result'), 'w'))


if __name__ == '__main__':
    # train()
    # test()
    ground_truth, proposals = pickle.load(open(os.path.join(THUMOS14.RES_DIR, 'detection_result'), 'r'))
    mAP(ground_truth, proposals)
