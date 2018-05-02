#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-1

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from data_provider.dataset import Dataset
from data_provider.utils import *
from video.flow_test import dense_optical_flow_test,LK_optical_flow


class Example(Dataset):
    """
    """

    def __init__(self, ):
        """Constructor for Example"""
        self.image = "dog.jpg"
        self.audio = "audio.wav"
        # self.video = "walking_dog.avi"
        # self.video = "horse_riding.avi"
        # self.video = "biking.avi"
        self.video = "rowing.avi"
        self.image = os.path.join(Dataset.EXAMPLE_DIR, self.image)
        self.audio = os.path.join(Dataset.EXAMPLE_DIR, self.audio)
        self.video = os.path.join(Dataset.EXAMPLE_DIR, self.video)

    def preprocess(self):
        frames_dir=os.path.join(Dataset.EXAMPLE_DIR,"frames")
        single_video_to_image(self.video,frames_dir,frame=4)
        flow_dir=os.path.join(Dataset.EXAMPLE_DIR,"flow")
        dense_optical_flow_test(self.get_frames(),flow_dir)
        flow_dir = os.path.join(Dataset.EXAMPLE_DIR, "flow")
        LK_optical_flow(self.get_frames(), flow_dir)


    def get_image(self):
        return [self.image]

    def get_video(self):
        return self.video
    def get_audio(self):
        return self.audio

    def get_frames(self,num=16):
        return get_file_list(os.path.join(self.EXAMPLE_DIR,'frames'))[:num]

    def get_flow(self,num=16):
        return get_file_list(os.path.join(self.EXAMPLE_DIR,'flow'))[:num]

    def get_video_name(self):
        return os.path.splitext(self.video)[0].split('/')[-1]


if __name__ == '__main__':
    example = Example()
    example.preprocess()