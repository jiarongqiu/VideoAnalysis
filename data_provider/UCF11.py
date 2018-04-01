#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/03/2018
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_provider.dataset import Dataset
from data_provider.utils import *
import random
import numpy as np


DIR_NAME = '/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio'
LABEL_LIST = '/Users/qiujiarong/Desktop/Video/data/ucf11.label.list'
TRAIN_LIST = '/Users/qiujiarong/Desktop/Video/data/ucf11.train.list'
TEST_LIST = '/Users/qiujiarong/Desktop/Video/data/ucf11.test.list'
NUM_CLASSES = 11
BATCH_SIZE = 16
NUM_PER_CLIP = 4


class UCF11(Dataset):

    def __init__(self):
        Dataset.__init__(self)
        self.load_in()
        self.idx = range(len(self.train_X))
        random.shuffle(self.idx)
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        self.dir_name=DIR_NAME

    def preprocess(self, frame=4):
        for category in get_dir_list(self.dir_name):
            for videos in get_dir_list(category, "Annotation"):
                for clip in get_file_list(videos):
                    single_video_to_image(clip, frame)

    def split(self):
        label_file = open(LABEL_LIST, 'w')
        train_file = open(TRAIN_LIST, 'w')
        test_file = open(TEST_LIST, 'w')
        cnt = 0
        for category in get_dir_list(DIR_NAME):
            label_file.write(os.path.split(category)[-1] + '\t' + str(cnt) + '\n')
            length = 0
            for videos in get_dir_list(category, "Annotation"):
                length += len(get_dir_list(videos))
            cnt2 = 0
            for videos in get_dir_list(category, "Annotation"):
                for clip in get_dir_list(videos):
                    if cnt2 < length * self.split_ratio:
                        train_file.write(clip + '\t' + str(cnt) + '\n')
                    else:
                        test_file.write(clip + '\t' + str(cnt) + '\n')
                    cnt2 += 1
            cnt += 1

    def load_in(self):
        with open(TRAIN_LIST, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                x = line[0]
                y = int(line[1])
                if (len(get_file_list(x)) < NUM_PER_CLIP): continue
                self.train_X.append(x)
                self.train_y.append(y)
        with open(TEST_LIST, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                x = line[0]
                y = int(line[1])
                if (len(get_file_list(x)) < NUM_PER_CLIP): continue
                self.test_X.append(x)
                self.test_y.append(y)

    def get_frames_data(self, clip):
        ret = []
        candidates = get_file_list(clip, sort=True)
        length = len(candidates)
        s_index = random.randint(0, length - self.num_frames_per_clip)
        for idx in range(s_index, s_index + self.num_frames_per_clip):
            ret.append(candidates[idx])
        return ret

    def get_next_batch(self):
        batch_X, batch_y = Dataset.get_next_batch(self)
        return batch_X, batch_y






if __name__ == '__main__':
    a = UCF11()
    a.preprocess()
    # a.split()
    # a.get_next_batch()
    # a.inception_v2_mean_embedding()
