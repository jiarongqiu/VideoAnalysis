import tensorflow as tf
import numpy as np
import turn_model
from six.moves import xrange
import time
import pickle
import os
from ActivityNet import ActivityNet
from sklearn.metrics import log_loss, accuracy_score
import math
from evaluate import cal_average_recall, cal_iou

ctx_num = 4
unit_size = 1
unit_feature_size = 2048
lr = 0.005
lambda_reg = 2.0
batch_size = 128
test_steps = 4000


def softmax(x):
    # print(x,math.log(np.sum(np.exp(x),axis=0)),np.exp(x-math.log(np.sum(np.exp(x),axis=0))))
    return np.exp(x - math.log(np.sum(np.exp(x), axis=0)))


def training():
    initial_steps = 0
    max_steps = 20000

    model = turn_model.TURN_Model()

    with tf.Graph().as_default():

        loss_cls_reg, vs_train_op, vs_eval_op, loss_reg = model.construct_model()
        tf.summary.scalar('loss', loss_cls_reg)
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs/", sess.graph)

        start_time = time.time()
        for step in xrange(max_steps):
            feed_dict = model.fill_feed_dict_train_reg()
            _, loss_v, loss_reg_v = sess.run([vs_train_op, loss_cls_reg, loss_reg], feed_dict=feed_dict)

            if step % 50 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                pt,epoch,train_set_length=model.train_set.get_training_info()
                # rs = sess.run(merged)
                # writer.add_summary(rs, step)
                # Print status to stdout.
                print('Step %d (%d/%d) Epoch %d  total loss = %.2f, regression loss = %.2f(%.3f sec)' % (
                    step, pt,train_set_length,epoch,loss_v, loss_reg_v, duration))
            if step % 1000 == 0:
                probs = []
                labels = []
                pred = []
                for i in range(1000):
                # for i in range(model.val_set):
                    movie_name, gt_start, gt_end, clip_start, clip_end, feat,unit_duration = model.val_set.get_sample(i)
                    feed_dict = {model.visual_featmap_ph_test: feat}
                    outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
                    pred_start, pred_end, action_score = postprocess(clip_start, clip_end, outputs,unit_duration)
                    # print(movie_name, gt_start, gt_end, pred_start, pred_end, action_score)
                    if gt_start == gt_end == 0:
                        label = 0
                    else:
                        label = 1
                    if action_score > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)
                    probs.append(action_score)
                    labels.append(label)
                # print(pred,probs,labels)
                print("Accuracy", accuracy_score(y_pred=pred, y_true=labels))
                print("Loss", log_loss(y_pred=probs, y_true=labels))
        saver.save(sess, os.path.join(ActivityNet.MODEL_DIR, 'turn_activity_net.model'))


def postprocess(round_start, round_end, outputs,unit_duration):
    pred_start = max(round_start - outputs[2] * unit_duration, 0)
    pred_end = round_end - outputs[3] * unit_duration
    action_score = softmax(outputs[:2])[1]
    if action_score == 0:
        action_score = 1e-8
    elif action_score == 1:
        action_score = 0.9999
    return pred_start, pred_end, action_score

def validation():
    model = turn_model.TURN_Model()

    with tf.Graph().as_default():
        sess = tf.Session()

        # First let's load meta graph and restore weights
        # saver = tf.train.import_meta_graph(os.path.join(THUMOS14.MODEL_DIR,'turn_tap_model.meta'))
        loss_cls_reg, vs_train_op, vs_eval_op, loss_reg = model.construct_model(test=True)
        # init = tf.global_variables_initializer()
        # sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ActivityNet.MODEL_DIR, 'turn_activity_net.model'))
        # utils.show_tensor(os.path.join(ActivityNet.MODEL_DIR, 'turn_tap_model'))

        result = {}
        probs = []
        labels = []
        pred = []
        for i in range(len(model.val_set.samples)):
            if i%1000==0:print(i,len(model.val_set.samples))
            movie_name, gt_start, gt_end, clip_start, clip_end, feat,unit_duration = model.val_set.get_sample(i)
            feed_dict = {
                model.visual_featmap_ph_test: feat
            }
            outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
            pred_start, pred_end, action_score = postprocess(clip_start, clip_end, outputs,unit_duration)
            # print(movie_name,gt_start,gt_end,pred_start,pred_end,action_score)
            _list = result.get(movie_name, [])
            _list.append((pred_start, pred_end, action_score))
            result[movie_name] = _list
            if gt_start == gt_end == 0:
                label = 0
            else:
                label = 1
            if action_score > 0.5:
                pred.append(1)
            else:
                pred.append(0)
            probs.append(action_score)
            labels.append(label)

        # print(pred,probs,labels)
        print("Accuracy", accuracy_score(y_pred=pred, y_true=labels))
        print("Loss", log_loss(y_pred=probs, y_true=labels))
    print(len(result))
    pickle.dump(result, open(os.path.join(ActivityNet.RES_DIR, 'turn_activity_net.result'), 'w'))


def eval():
    def filter(proposals, tiou=0.95):
        removed = []
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                if j in removed: continue
                if cal_iou(proposals[i][:2], proposals[j][:2]) > tiou:
                    removed.append(i)
                    break
        return sorted([x for idx, x in enumerate(proposals) if idx not in removed], key=lambda x: x[2], reverse=True)

    dataset = ActivityNet()
    val_dataset =dataset.val_info

    result = pickle.load(open(os.path.join(ActivityNet.RES_DIR, 'turn_activity_net.result'), 'r'))
    print(len(result),len(val_dataset))
    ret = {}
    for movie in val_dataset:
        proposals = result.get(movie,[])
        # proposals = sorted(list(proposals), key=lambda x: (x[0], x[1]))
        proposals = filter(proposals)
        proposals = [{"start": start, "end": end, "score": score} for start, end, score in proposals]
        ret[movie] = {'proposals': proposals}
    pickle.dump(ret, open(os.path.join(ActivityNet.RES_DIR, 'turn_activity_net_proposal'), 'w'))
    print(cal_average_recall(predicts=ret, groundtruth=val_dataset, num_proposals=200))


def main(_):
    # training()
    validation()
    # validation()
    # eval()


if __name__ == '__main__':
    tf.app.run()
