#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/04/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np
from video.utils import show_tensor
import PIL.Image as Image
import cv2
import json
from video.c3d_model import C3DModel
from data_provider.example import Example
from data_provider.UCF101 import UCF101
from sklearn.decomposition import IncrementalPCA

MODEL_DIR = Example.MODEL_DIR
EXAMPLE_DIR = Example.EXAMPLE_DIR
class C3D(object):
    """"""
    num_frames_per_clip = 16
    crop_size = 112
    batch_size=1

    def __init__(self, ):
        """Constructor for C3D"""
        # mean file slightly differences
        self.np_mean = np.load(os.path.join(MODEL_DIR, 'crop_mean.npy')).reshape( [self.num_frames_per_clip, self.crop_size, self.crop_size, 3])
        self.model=C3DModel(batch_size=self.batch_size,test=True)
        self.checkpoint_file = os.path.join(MODEL_DIR, 'c3d_ucf101_finetune_whole_iter_20000_TF.model')
        self.input_tensor,_=self.model.placeholder_inputs()
        self.out,self.end_points=self.model.inference_c3d(self.input_tensor)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)
        self.sess = sess
        print("C3D Model Load In Finished")

    def test(self,frames):
        frames = self.read_clip(frames)
        frames=frames.reshape(1,self.num_frames_per_clip,self.crop_size,self.crop_size,3)
        out,ret=self.sess.run([self.out,self.end_points], feed_dict={self.input_tensor: frames})
        idx2label={}
        with open(UCF101.LABEL_LIST,'r') as fr:
            for line in fr:
                label,idx=line.strip().split()
                idx=int(idx)
                idx2label[idx]=label
        print("result:",idx2label[np.argmax(out)],"score:",out[0,np.argmax(out)])
        feature1=ret['fc_6']
        feature2=ret['fc_7']
        print("FC_6",feature1.shape,feature1[0,:10])
        print("FC_7",feature2.shape,feature2[0,:10])

    def predict(self,frames):
        frames=self.read_clip(frames)
        frames = frames.reshape(1, self.num_frames_per_clip, self.crop_size, self.crop_size, 3)
        out = self.sess.run(self.out, feed_dict={self.input_tensor: frames})
        return np.argmax(out)

    def get_feature(self,frames,layer='fc_7'):
        '''
        :param frames:
        :return: 4096 dims features
        '''
        frames=self.read_clip(frames)
        frames = frames.reshape(1, self.num_frames_per_clip, self.crop_size, self.crop_size, 3)
        if layer=='fc_7':
            out, ret = self.sess.run([self.out, self.end_points['fc_7']], feed_dict={self.input_tensor: frames})
        elif layer=='fc_6':
            out, ret = self.sess.run([self.out, self.end_points['fc_6']], feed_dict={self.input_tensor: frames})
        return ret.reshape(-1)


    def read_clip(self,frames):
        def get_frames_data(frames):
            ''' Given a directory containing extracted frames, return a video clip of
            (num_frames_per_clip) consecutive frames as a list of np arrays '''
            ret_arr = []
            for image_name in frames:
                img = Image.open(image_name)
                img_data = np.array(img)
                ret_arr.append(img_data)
            return ret_arr
        assert len(frames)==self.num_frames_per_clip
        tmp_data= get_frames_data(frames)
        data=[]
        for j in xrange(len(tmp_data)):
            img = Image.fromarray(tmp_data[j].astype(np.uint8))
            if (img.width > img.height):
                scale = float(self.crop_size) / float(img.height)
                img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), self.crop_size))).astype(
                    np.float32)
            else:
                scale = float(self.crop_size) / float(img.width)
                img = np.array(cv2.resize(np.array(img), (self.crop_size, int(img.height * scale + 1)))).astype(
                    np.float32)
            crop_x = int((img.shape[0] - self.crop_size) / 2)
            crop_y = int((img.shape[1] - self.crop_size) / 2)
            img = img[crop_x:crop_x + self.crop_size, crop_y:crop_y + self.crop_size, :] - self.np_mean[j]
            data.append(img)
        np_arr_data = np.array(data).astype(np.float32)
        return np_arr_data


    def c3d_fc6_feature(self):
        FEATURE_DIR=os.path.join(UCF101.FEATURE_DIR,'c3d_fc_6')
        dataset=UCF101()
        dataset.load_in()
        test_list = dataset.test_list
        train_list = dataset.train_list
        for video, label in train_list:
            name=video.split('/')[-1]
            path=os.path.join(FEATURE_DIR,name+'.np')
            feature=self.get_feature(dataset.get_random_clip(video),layer='fc_6')
            feature.dump(path)
        for video, label in test_list:
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            feature = self.get_feature(dataset.get_random_clip(video),layer='fc_6')
            feature.dump(path)
    def c3d_fc7_feature(self):
        FEATURE_DIR=os.path.join(UCF101.FEATURE_DIR,'c3d_fc_7')
        dataset=UCF101()
        dataset.load_in()
        test_list = dataset.test_list
        train_list = dataset.train_list
        for video, label in train_list:
            name=video.split('/')[-1]
            path=os.path.join(FEATURE_DIR,name+'.np')
            feature=self.get_feature(dataset.get_random_clip(video),layer='fc_7')
            feature.dump(path)
        for video, label in test_list:
            name = video.split('/')[-1]
            path = os.path.join(FEATURE_DIR, name + '.np')
            feature = self.get_feature(dataset.get_random_clip(video),layer='fc_7')
            feature.dump(path)
def main():
    c3d=C3D()
    # example = Example()
    # c3d.test(example.get_frames())
    # c3d.test(example.get_flow())
    # c3d.test(c3d.read_clip(example.get_frames()))
    # print(example.get_frames())
    # c3d.get_feature(example.get_frames())
    # c3d.c3d_fc6_feature()
    # c3d.c3d_fc7_feature()
    # show_tensor(c3d.checkpoint_file)


if __name__ == '__main__':
    main()