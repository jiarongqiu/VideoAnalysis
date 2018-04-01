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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import PIL.Image as Image
import random
import numpy as np


def get_dir_list(path, _except=None):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and x != _except]


def get_file_list(path, _except=None, sort=False):
    if sort:
        return sorted(
            [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and x != _except])
    else:
        return [os.path.join(path, x) for x in os.listdir(path) if
                os.path.isfile(os.path.join(path, x)) and x != _except]


class dataReader():

    def __init__(self, filename, batch_size=16, num_frames_per_clip=16):
        self.start_pos = 0
        self.batch_size = batch_size
        self.num_frames_per_clip = num_frames_per_clip
        self.clips = []
        self.labels = []
        with open(filename, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                clip = line[0]
                label = int(line[1])
                if (len(get_file_list(clip)) < num_frames_per_clip): continue
                self.clips.append(clip)
                self.labels.append(label)
        self.size = len(self.clips)
        self.idx = range(self.size)
        random.shuffle(self.idx)
        self.finished = False

    def get_frames_data(self, clip):
        ret = []
        candidates = get_file_list(clip, sort=True)
        length = len(candidates)
        s_index = random.randint(0, length - self.num_frames_per_clip)
        for idx in range(s_index, s_index + self.num_frames_per_clip):
            img = Image.open(candidates[idx])
            ret.append(np.array(img))
        return ret

    def get_next_batch(self):
        data = []
        labels = []
        for i in range(self.batch_size):
            if(self.start_pos>=self.size):
                self.start_pos = 0
                random.shuffle(self.idx)
                self.finished = True
            clip = self.clips[self.idx[self.start_pos ]]
            label = self.labels[self.idx[self.start_pos ]]
            data.append(self.get_frames_data(clip))
            labels.append(label)
            self.start_pos+=1
        data = np.array(data).astype(np.float32)
        labels = np.array(labels).astype(np.int64)

        return data, labels

    def is_finished(self):
        return self.finished
    def get_size(self):
        return self.size


if __name__ == '__main__':
    train = dataReader("ucf-101_train_list.txt")
    data, label = train.get_next_batch()
    print(data.shape, label.shape)
    print(label)
