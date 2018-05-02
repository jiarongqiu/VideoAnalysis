#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 31/03/2018

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import os

from data_provider.example import Example
from audio import vggish_postprocess, vggish_slim, vggish_params, vggish_input
import numpy as np
slim = tf.contrib.slim

MODEL_DIR = Example.MODEL_DIR
DEMO_DIR = Example.EXAMPLE_DIR
example=Example()

class VGG(object):
    """
        Vggish Audio CNN
    """

    def __init__(self):
        """Constructor for AudioFeature"""
        checkpoint_path = os.path.join(MODEL_DIR, 'vggish_model.ckpt')
        pca_params_path = os.path.join(MODEL_DIR, 'vggish_pca_params.npz')
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        sess = tf.Session()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        self.embedding_tensor = embedding_tensor
        self.features_tensor = features_tensor
        self.sess = sess
        self.pproc = pproc
        print("VGG Model Load In Finished")

    def wav2feature(self, wav_file):
        if not os.path.exists(wav_file):
            feat = np.zeros(128)
        else:
            examples_batch = vggish_input.wavfile_to_examples(wav_file)
            [embedding_batch] = self.sess.run([self.embedding_tensor], feed_dict={self.features_tensor: examples_batch})
            postprocessed_batch = self.pproc.postprocess(embedding_batch)
            feat = np.mean(postprocessed_batch, axis=0)
        return feat

    def test(self):
        wav_file = example.get_audio()
        print (self.wav2feature(wav_file).shape)
        print("VGG Test Finished")


def main():
    vgg = VGG()
    vgg.test()


if __name__ == '__main__':
    main()
