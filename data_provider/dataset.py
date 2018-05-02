#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/03/2018
"""base class of all dataset"""
import random


class Dataset:
    num_classes = 0
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    split_ratio = 0.3
    start_pos = 0
    batch_size = 0
    finished = False
    MODEL_DIR = "/home/charlie/Desktop/VideoAnalysis/models"
    EXAMPLE_DIR = "/home/charlie/Desktop/VideoAnalysis/examples"
    def __init__(self):

        pass

    # offline
    def load_in(self):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass

    def split(self):
        pass

    # online
    def get_data(self, split_name):
        if split_name == 'train':
            return self.train_X, self.train_y
        elif split_name == 'test':
            return self.test_X, self.test_y
        else:
            raise ValueError("train/test unclear")

    def get_next_batch(self):
        batch_X = []
        batch_y = []
        for i in range(self.batch_size):
            if (self.start_pos >= self.num_samples):
                self.start_pos = 0
                random.shuffle(self.idx)
                self.finished = True
            x = self.X[self.idx[self.start_pos]]
            label = self.y[self.idx[self.start_pos]]
            batch_X.append(x)
            batch_y.append(label)
            self.start_pos += 1
        return batch_X, batch_y

    def is_finished(self):
        return self.finished

    def random_drop(self, _list, num):
        idx = range(len(_list))
        random.shuffle(idx)
        return _list[idx[:num]]


if __name__ == '__main__':
    a = Dataset()
    print(a.num_classes)
