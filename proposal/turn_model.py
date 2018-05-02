import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from proposal import vs_multilayer
from proposal.cnn import fc_layer as fc
from dataset import TestingDataSet
from dataset import TrainingDataSet

class TURN_Model(object):
    ctx_num = 4
    unit_size = 16.0
    unit_feature_size = 4096
    lr = 0.005
    lambda_reg = 2.0
    batch_size = 128
    test_steps = 4000
    test_batch_size=1
    visual_feature_dim=unit_feature_size*3
    def __init__(self,):
        self.train_set = TrainingDataSet(self.batch_size)
        self.test_set = TestingDataSet()

    def fill_feed_dict_train_reg(self):
        image_batch, label_batch, offset_batch = self.train_set.next_batch()
        input_feed = {
            self.visual_featmap_ph_train: image_batch,
            self.label_ph: label_batch,
            self.offset_ph: offset_batch
        }
        return input_feed

    # construct the top network and compute loss
    def compute_loss_reg(self, visual_feature, offsets, labels,test=False):

        cls_reg_vec = vs_multilayer.vs_multilayer(visual_feature, "APN", middle_layer_dim=1000,test=test)
        cls_reg_vec = tf.reshape(cls_reg_vec, [self.batch_size, 4])
        cls_score_vec_0, cls_score_vec_1, p_reg_vec, l_reg_vec = tf.split(cls_reg_vec, 4, 1)
        cls_score_vec = tf.concat((cls_score_vec_0, cls_score_vec_1), 1)
        offset_pred = tf.concat((p_reg_vec, l_reg_vec), 1)
        # classification loss
        loss_cls_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_vec, labels=labels)
        loss_cls = tf.reduce_mean(loss_cls_vec)
        # regression loss
        label_tmp = tf.to_float(tf.reshape(labels, [self.batch_size, 1]))
        label_for_reg = tf.concat([label_tmp, label_tmp], 1)
        loss_reg = tf.reduce_mean(tf.multiply(tf.abs(tf.subtract(offset_pred, offsets)), label_for_reg))

        loss = tf.add(tf.multiply(self.lambda_reg, loss_reg), loss_cls)
        return loss, offset_pred, loss_reg

    def init_placeholder(self):
        visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim))
        label_ph = tf.placeholder(tf.int32, shape=(self.batch_size))
        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim))
        # print(visual_featmap_ph_train, visual_featmap_ph_test, label_ph, offset_ph)
        return visual_featmap_ph_train, visual_featmap_ph_test, label_ph, offset_ph

    # set up the eval op
    def eval(self, visual_feature_test):
        # visual_feature_test=tf.reshape(visual_feature_test,[1,4096])
        outputs = vs_multilayer.vs_multilayer(visual_feature_test, "APN", middle_layer_dim=1000, reuse=True)
        outputs = tf.reshape(outputs, [4])
        return outputs

    # return all the variables that contains the name in name_list
    def get_variables_by_name(self, name_list):
        v_list = tf.trainable_variables()
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print "Variables of <" + name + ">"
            for v in v_dict[name]:
                print "    " + v.name
        return v_dict

    # set up the optimizer
    def training(self, loss):
        v_dict = self.get_variables_by_name(["APN"])
        vs_optimizer = tf.train.AdamOptimizer(self.lr, name='vs_adam')
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["APN"])
        return vs_train_op

    # construct the network
    def construct_model(self,test=False):
        self.visual_featmap_ph_train, self.visual_featmap_ph_test, self.label_ph, self.offset_ph = self.init_placeholder()
        visual_featmap_ph_train_norm = tf.nn.l2_normalize(self.visual_featmap_ph_train, axis=1)
        visual_featmap_ph_test_norm = tf.nn.l2_normalize(self.visual_featmap_ph_test, axis=1)
        self.loss_cls_reg, offset_pred, loss_reg = self.compute_loss_reg(visual_featmap_ph_train_norm, self.offset_ph,
                                                                      self.label_ph,test)
        self.train_op = self.training(self.loss_cls_reg)
        eval_op = self.eval(visual_featmap_ph_test_norm)
        return self.loss_cls_reg, self.train_op, eval_op, loss_reg


if __name__ == '__main__':
    model=TURN_Model()
    model.construct_model()
    print(model.fill_feed_dict_train_reg())