#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 08/03/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from PIL import Image

from video.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from audio import vggish_postprocess, vggish_slim, vggish_params, vggish_input
from video.flow_test import dense_optical_flow

slim = tf.contrib.slim

DIR_NAME = "../data/"


class RGBFeature(object):
    """"""

    def __init__(self):
        """Constructor for RGBFeature"""
        checkpoint_file = DIR_NAME + 'checkpoints/inception_resnet_v2_2016_08_30.ckpt'
        self.input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
        sess = tf.Session()
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            _, end_points = inception_resnet_v2(self.input_tensor, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        self.end_points = end_points
        self.sess = sess
        print("RGB Model Load In Finished")

    def images2feature(self, images):
        ret = []
        for image in images:
            im = Image.open(image).resize((299, 299))
            im = np.array(im)
            im = im.reshape(-1, 299, 299, 3)
            im = 2 * (im / 255.0) - 1.0
            drop_out_values = self.sess.run(self.end_points['PreLogitsFlatten'], feed_dict={self.input_tensor: im})
            # print(drop_out_values.shape)
            ret.append(drop_out_values.squeeze())
        return np.array(ret)

    def test(self):
        sample_images = [DIR_NAME + 'dog.jpg']
        print(self.images2feature(sample_images).shape)


class FlowFeature(object):
    """"""

    def __init__(self ):
        """Constructor for FlowFeature"""

    def images2feature(self,images):
        return dense_optical_flow(images)

    def test(self):
        sample_images = []
        for i in range(1,6):
            sample_images.append(DIR_NAME+"flow_test/"+str(i)+".jpg")
        print(sample_images)
        print(self.images2feature(images=sample_images).shape)



class AudioFeature(object):
    """"""

    def __init__(self):
        """Constructor for AudioFeature"""
        checkpoint_path = DIR_NAME + 'checkpoints/vggish_model.ckpt'
        pca_params_path = DIR_NAME + 'checkpoints/vggish_pca_params.npz'
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        sess = tf.Session()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        self.embedding_tensor = embedding_tensor
        self.features_tensor = features_tensor
        self.sess = sess
        self.pproc = pproc
        print("Audio Model Load In Finished")

    def wav2feature(self, wav_file):
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: examples_batch})
        postprocessed_batch = self.pproc.postprocess(embedding_batch)
        return postprocessed_batch

    def test(self):
        wav_file = DIR_NAME + "audio_test.wav"
        print(self.wav2feature(wav_file).shape)


if __name__ == '__main__':
    # rgb=RGBFeature()
    # rgb.test()
    audio = AudioFeature()
    audio.test()
    # flow=FlowFeature()
    # flow.test()
