#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by charlie on 18-4-1

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from data_provider.dataset import Dataset
from data_provider.utils import *

class Example(Dataset):
    """
    """

    def __init__(self, ):
        """Constructor for Example"""
        self.image = "dog.jpg"
        self.audio = "audio.wav"
        self.video = "walking_dog.avi"
        self.image = os.path.join(Dataset.EXAMPLE_DIR, self.image)
        self.audio = os.path.join(Dataset.EXAMPLE_DIR, self.audio)
        self.video = os.path.join(Dataset.EXAMPLE_DIR, self.video)

    def preprocess(self):
        folder = get_parent_folder(self.image)
        single_video_to_image(self.image,folder)

    def get_image(self):
        return self.image


if __name__ == '__main__':
    example = Example()
    example.preprocess()
