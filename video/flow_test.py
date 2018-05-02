#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 08/03/2018

import cv2
import numpy as np
import os


def dense_optical_flow_test(frames,folder):
    os.system("rm -rf %s" % folder)
    os.system("mkdir %s" % folder)
    # print(frames,folder)
    assert len(frames) > 1
    pt = 0
    frame1 = cv2.imread(frames[pt])
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    pt += 1
    while (pt != len(frames)):
        frame2 = cv2.imread(frames[pt])
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(folder,"dense_flow_" + str(pt) + ".png"),rgb)
        prvs = next
        pt += 1


def dense_optical_flow(images):
    assert len(images) > 1
    ret = []
    pt = 0
    frame1 = cv2.imread(images[pt])
    print(frame1.shape)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    pt += 1
    while (pt != len(images)):
        frame2 = cv2.imread(images[pt])
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ret.append(rgb)
        prvs = next
        pt += 1
    return np.array(ret)


def LK_optical_flow(frames, folder):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    pt = 0
    frame1 = cv2.imread(frames[pt])
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)
    pt += 1

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)

    while (pt != len(frames)):
        frame2 = cv2.imread(frames[pt])
        print(frame2.shape)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

        print(p1.shape, st.shape)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            cv2.circle(frame2, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame2, mask)
        cv2.imwrite(os.path.join(folder,"LK_flow" + str(pt) + ".png"), mask)
        prvs = next
        pt += 1


def main():
    images = ['../data/flow_test/1.jpg', '../data/flow_test/2.jpg', '../data/flow_test/3.jpg',
              '../data/flow_test/4.jpg', '../data/flow_test/5.jpg']
    dense_optical_flow(images)
    # LK_optical_flow(images)

if __name__ == '__main__':
    main()
