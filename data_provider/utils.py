#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: qiujiarong
# Date: 01/03/2018

import os
import subprocess

def get_dir_list(path, _except=None):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and x != _except]


def get_file_list(path, _except=None, sort=True):
    if sort:
        return sorted(
            [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and x != _except])
    else:
        return [os.path.join(path, x) for x in os.listdir(path) if
                os.path.isfile(os.path.join(path, x)) and x != _except]

def create_folder(folder):
    os.system("rm -rf %s" % folder)
    os.system("mkdir %s" % folder)


def single_video_to_image(infile,folder, frame=5, crop_size=299):
    os.system("rm -rf %s" % folder)
    os.system("mkdir %s" % folder)
    str = "ffmpeg -i %s -s %d*%d -vf fps=%d %s" % (infile, crop_size, crop_size, frame, folder) + "/%05d.jpg"
    os.system(str)

def single_video_to_wav(infile,outfile):
    # outfile=os.path.splitext(infile)[0]+".wav"
    str = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn %s"%(infile,outfile)
    # print(str)
    os.system("rm %s"%(outfile))
    os.system(str)

def get_video_duration(file):
    # os.system("ffmpeg -i "+file+" 2>&1 | grep Duration | awk '{print $2}' | tr -d ,")
    cmd="ffprobe -i "+file+" -show_entries format=duration -v quiet -of csv='p=0'"
    time=float(subprocess.check_output(cmd, shell=True))
    return time

if __name__ == '__main__':
    print(get_video_duration("../examples/walking_dog.avi"))
    # single_video_to_wav("/Users/qiujiarong/Desktop/Video/data/test.avi")
    # print(get_file_list("/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/"))
    # print(get_file_list("/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01"))
    # single_video_to_wav("/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01.avi")
    # single_video_to_image("/Users/qiujiarong/Desktop/Video/data/action_youtube_naudio/volleyball_spiking/v_spiking_12/v_spiking_12_01.avi")