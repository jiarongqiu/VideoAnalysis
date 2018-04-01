# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from data import dataReader
from model import C3D

# Basic model parameters as external flags.
flags = tf.app.flags
# flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 500, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './out'
NUM_FRAMES_PER_CLIP = 16
CROP_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 16


def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_FRAMES_PER_CLIP,CROP_SIZE,CROP_SIZE,CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
    return images_placeholder, labels_placeholder


def get_loss_accuracy(logit,labels):
    pred=tf.argmax(logit, 1)
    correct_pred = tf.equal(pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit))
    return loss,accuracy,pred

def train(dataset="UCF-101"):
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    dataset=dataset.lower()
    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # use_pretrained_model = True
    # model_filename = "./sports1m_finetuning_ucf101.model"


    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs()
        lr = tf.placeholder(tf.float32)
        c3d = C3D.C3D(num_class=101)

        logit = c3d.model(images_placeholder, batch_size=BATCH_SIZE)
        loss,accuracy,pred=get_loss_accuracy(logit,labels_placeholder)

        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        train_data = dataReader.dataReader("data/"+dataset+"_train_list.txt", batch_size=BATCH_SIZE)
        test_data = dataReader.dataReader("data/"+dataset+"_test_list.txt", batch_size=BATCH_SIZE)
        init = tf.global_variables_initializer()
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        )
        sess.run(init)
        saver = tf.train.Saver()
        # ckpt=tf.train.get_checkpoint_state(model_save_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     print(ckpt.model_checkpoint_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # Create a session for running Ops on the Graph.

        learning_rate=0.003
        for step in range(FLAGS.max_steps):
            if step!=0 and step%2e+5 ==0:learning_rate/=10
            # start_time = time.time()
            train_images, train_labels = train_data.get_next_batch()

            sess.run(train_op, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels,
                lr:learning_rate
            })
            # duration = time.time() - start_time
            # print('Step %d: %.3f sec' % (step, duration))

            # Save a checkpoint and evaluate the model periodically.
            if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
                print('step %d Train'%(step))
                acc, _loss, _pred,_logit = sess.run(
                    [accuracy, loss, pred,logit],
                    feed_dict={images_placeholder: train_images,
                               labels_placeholder: train_labels,
                               lr: learning_rate
                               })

                print (_logit)
                # print (_pred)
                print ("accuracy: " + "{:.5f}".format(acc) + " loss:" + "{:.5f}".format(_loss))
            if (step) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                print('step %d Test'%(step))
                test_images, test_labels = test_data.get_next_batch()
                _loss, acc = sess.run(
                    [loss, accuracy],
                    feed_dict={
                        images_placeholder: test_images,
                        labels_placeholder: test_labels,
                        lr: learning_rate
                    })
                print ("accuracy: " + "{:.5f}".format(acc) + " loss:" + "{:.5f}".format(_loss))
        saver.save(sess,model_save_dir+'/'+dataset+'.model')

    print("done")

def test(dataset="UCF-101"):
    dataset=dataset.lower()
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs()
        c3d = C3D.C3D(num_class=101)

        logit = c3d.model(images_placeholder, batch_size=BATCH_SIZE)
        correct_pred = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit, 1), labels_placeholder),tf.int16))
        print(correct_pred)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_save_dir + '/'+dataset+'.model')
        test_data = dataReader.dataReader("data/"+dataset+"_test_list.txt", batch_size=BATCH_SIZE)
        finished=False
        cnt=0
        total=0
        while(not finished):
            test_images, test_labels = test_data.get_next_batch()
            finished=test_data.is_finished()
            _correct_pred = sess.run(
                [correct_pred],
                feed_dict={
                    images_placeholder: test_images,
                    labels_placeholder: test_labels
                })
            total+=test_images.shape[0]
            cnt+=_correct_pred[0]
        print(cnt,total,cnt/float(total))


def main(_):
    train()
    # test()


if __name__ == '__main__':
    tf.app.run()
