#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-17
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import random
import pickle

from data_provider.THUMOS14 import THUMOS14
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import time
import tensorflow as tf
from proposal.evaluate import cal_iou, cal_average_recall


class DataSetUtils(object):
    def __init__(self, batch_size=256):
        self.dataset = THUMOS14()
        self.batch_size = batch_size
        self.visual_feature_dim = 4096
        self.feat_dir = self.dataset.FEATURE_DIR
        self.unit_duration = 1.0
        self.training_samples, self.testing_sample = self.prepare_sample()
        print(len(self.training_samples), len(self.testing_sample))
        self.idx = 0

    def prepare_sample(self):
        samples = []
        train, test = self.dataset.load_in_info()
        for vid in train:
            used = set()
            info = train[vid]
            proposals = info['proposals']
            movie_name = vid
            for proposal in proposals:
                start = proposal['start']
                end = proposal['end']
                round_start = np.floor(start / self.unit_duration) * self.unit_duration
                round_end = np.floor(end / self.unit_duration) * self.unit_duration
                if round_start == round_end:
                    samples.append((movie_name, int(round_start), 1))
                    used.add(int(round_start))
                else:
                    while round_start <= round_end:
                        samples.append((movie_name, int(round_start), 1))
                        used.add(int(round_start))
                        round_start += self.unit_duration
            clip_start = 0
            clip_end = clip_start + self.unit_duration
            while clip_end < info['duration']:
                if int(clip_start) not in used:
                    samples.append((movie_name, int(clip_start), 0))
                clip_start += self.unit_duration
                clip_end = clip_start + self.unit_duration
        train_samples = sorted(samples)
        samples = []
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
                if round_start == round_end:
                    samples.append((movie_name, int(round_start), 1))
                    used.add(int(round_start))
                else:
                    while round_start <= round_end:
                        samples.append((movie_name, int(round_start), 1))
                        used.add(int(round_start))
                        round_start += self.unit_duration
            clip_start = 0
            clip_end = clip_start + self.unit_duration
            while clip_end < info['duration']:
                if int(clip_start) not in used:
                    samples.append((movie_name, int(clip_start), 0))
                clip_start += self.unit_duration
                clip_end = clip_start + self.unit_duration
        test_samples = sorted(samples)
        return train_samples, test_samples

    def next_train_batch(self):
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        label_batch = np.zeros([self.batch_size, 2], dtype=np.int32)
        index = 0
        while index < self.batch_size:
            k = self.idx
            movie_name = self.training_samples[k][0]
            round_gt_start = self.training_samples[k][1]
            label = self.training_samples[k][2]
            feat = self.dataset.load_feature(movie_name, round_gt_start, round_gt_start + self.unit_duration, l2=True)
            image_batch[index, :] = feat
            # label_batch[index]=label
            if label == 1:
                label_batch[index, 0] = 0
                label_batch[index, 1] = 1
            else:
                label_batch[index, 0] = 1
                label_batch[index, 1] = 0
            index += 1
            self.idx += 1
            if self.idx >= len(self.training_samples): self.idx = 0

        return image_batch, label_batch

    def get_train(self):
        image_batch = []
        label_batch = []
        for i in range(len(self.training_samples)):
            k = i
            movie_name = self.training_samples[k][0]
            round_gt_start = self.training_samples[k][1]
            label = self.training_samples[k][2]
            feat = self.dataset.load_feature(movie_name, round_gt_start, round_gt_start + self.unit_duration)
            image_batch.append(feat)
            label_batch.append(label)
            if i > 2000: break
        image_batch = np.array(image_batch)
        label_batch = np.array(label_batch)
        return image_batch, label_batch

    def get_test(self):
        image_batch = []
        label_batch = []
        for i in range(len(self.testing_sample)):
            k = i
            movie_name = self.testing_sample[k][0]
            round_gt_start = self.testing_sample[k][1]
            label = self.testing_sample[k][2]
            feat = self.dataset.load_feature(movie_name, round_gt_start, round_gt_start + self.unit_duration)
            image_batch.append(feat)
            label_batch.append(label)
            if i > 200: break
        image_batch = np.array(image_batch)
        label_batch = np.array(label_batch)
        return image_batch, label_batch


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))
    # h = tf.nn.sigmoid(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.001))


def training():
    dataset = DataSetUtils()
    X = tf.placeholder("float", shape=[None, 4096])
    y = tf.placeholder("float", shape=[None, 2])
    w_h = init_weight([4096, 1000])
    w_o = init_weight([1000, 2])
    logit = model(X, w_h, w_o)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)) * 100
    train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)
    saver = tf.train.Saver()
    max_iter = 50000
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(max_iter):
            train_X, train_y = dataset.next_train_batch()
            loss, _ = sess.run([cost, train_op], feed_dict={X: train_X, y: train_y})
            if i % 500 == 0:
                print("Step", i, "Loss", loss)
            if i % 10000 == 0:
                prob, pred, labels = [], [], []
                for i in range(len(dataset.testing_sample)):
                    movie_name = dataset.testing_sample[i][0]
                    round_gt_start = dataset.testing_sample[i][1]
                    label = dataset.testing_sample[i][2]
                    feat = dataset.dataset.load_feature(movie_name, round_gt_start,round_gt_start + dataset.unit_duration, l2=True)
                    feat = feat.reshape(1, -1)
                    _logit = sess.run(logit, feed_dict={X: feat.reshape([1, -1])})
                    if softmax(_logit)[0, 1] > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)
                    prob.append(softmax(_logit)[0, 1])
                    labels.append(label)
                print("Accuracy", accuracy_score(y_pred=pred, y_true=labels))
                print("Loss", log_loss(y_pred=prob, y_true=labels))
        saver.save(sess, os.path.join(THUMOS14.MODEL_DIR, 'action_classifier_' + str(max_iter)))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


def testing():
    dataset = DataSetUtils()
    result = {}
    with tf.Graph().as_default():
        sess = tf.Session()

        X = tf.placeholder("float", shape=[None, 4096])
        y = tf.placeholder("float", shape=[None, 2])
        w_h = init_weight([4096, 1000])
        w_o = init_weight([1000, 2])
        logit = model(X, w_h, w_o)

        # First let's load meta graph and restore weights
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(THUMOS14.MODEL_DIR, 'action_classifier_50000'))

        pred, probs, labels = [], [], []
        for i in range(len(dataset.testing_sample)):
            if i % 1000 == 0: print(i)
            movie_name = dataset.testing_sample[i][0]
            round_gt_start = dataset.testing_sample[i][1]
            label = dataset.testing_sample[i][2]
            feat = dataset.dataset.load_feature(movie_name, round_gt_start, round_gt_start + dataset.unit_duration,l2=True)
            feat = feat.reshape(1, -1)
            _logit = sess.run(logit, feed_dict={X: feat.reshape([1, -1])})
            if softmax(_logit)[0, 1] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
            prob = softmax(_logit)[0, 1]
            probs.append(prob)
            labels.append(label)
            _list = result.get(movie_name, [])
            _list.append((round_gt_start, prob))
            result[movie_name] = _list
        print("Accuracy", accuracy_score(y_pred=pred, y_true=labels))
        print("Loss", log_loss(y_pred=probs, y_true=labels))
    pickle.dump(result, open(os.path.join(THUMOS14.RES_DIR, 'tag_time_probs'), 'w'))


def postprocess():
    '''
    result:movie:[(timestamp,action_prob)]
    gamma:

    :return:
    '''

    def compensate(pred):
        ret = []
        for time, prob in pred:
            ret.append((time, 1 - prob))
        return sorted(ret, key=lambda x: x[0])

    def G(pred, gamma):
        candidates = []
        pre = 1
        if pred[0][1] < gamma:
            start = 0
        else:
            start = -1
        end = -1
        idx = 0
        probs=[]
        while idx < len(pred):
            time, prob = pred[idx]
            if start!=-1:probs.append(prob)
            if prob < gamma and pre > prob and end == -1: start = time
            if prob > gamma and pre < prob and start != -1:
                end = time
                candidates.append((start, end,sum(probs)/len(probs)))
                start = end = -1
                probs=[]
            pre = prob
            idx += 1
        # print(candidates)
        return candidates

    def absorb(candidates, tau):
        proposals = []
        for i in range(len(candidates)):
            start = candidates[i][0]
            end0=end = candidates[i][1]
            prob =candidates[i][2]
            for j in range(i + 1, len(candidates)):
                if candidates[j][0]<end:continue
                # print(candidates[j][1],(end0 - start) / (candidates[j][1] - start),tau)
                if (end0 - start) / (candidates[j][1] - start) < tau: break
                end = candidates[j][1]
            proposals.append((start, end,prob))
        return proposals

    def filter(proposals, tiou=0.95):
        removed = []
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                if j in removed: continue
                if cal_iou(proposals[i][:2], proposals[j][:2]) > tiou:
                    removed.append(i)
                    break
        return sorted([(x[0],x[1],1-x[2]) for idx, x in enumerate(proposals) if idx not in removed],key=lambda x:x[2],reverse=True)

    dataset = DataSetUtils()
    _, test = dataset.dataset.load_in_info()

    result = pickle.load(open(os.path.join(THUMOS14.RES_DIR, 'tag_time_probs'), 'r'))
    ret = {}
    for movie in test:
        pred = result[movie]
        pred = compensate(pred)
        proposals = set()
        for gamma in np.arange(0.05, 1, 0.05):
            G_set = G(pred, gamma)
            for tau in np.arange(0.05, 1, 0.05):
                candidates = absorb(G_set, tau)
                proposals = proposals.union(set(candidates))
        proposals = sorted(list(proposals), key=lambda x: (x[0], x[1]))
        proposals = filter(proposals)
        # print(sorted(proposals, key=lambda x: (x[0], x[1])))
        proposals = [{"start": start, "end": end} for start, end,_ in proposals]
        ret[movie] = {'proposals': proposals}
        # print(result[movie])
        # print(test[movie]['proposals'])
        # break
    print(cal_average_recall(predicts=ret, groundtruth=test, num_proposals=200))
    pickle.dump(ret, open(os.path.join(THUMOS14.RES_DIR, 'tag_proposal'), 'w'))



if __name__ == '__main__':
    # training()
    # testing()
    postprocess()
