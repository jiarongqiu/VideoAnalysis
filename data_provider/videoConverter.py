import os


def get_dir_list(path, _except=None):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and x != _except]


def get_file_list(path, _except=None, sort=False):
    if sort:
        return sorted(
            [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and x != _except])
    else:
        return [os.path.join(path, x) for x in os.listdir(path) if
                os.path.isfile(os.path.join(path, x)) and x != _except]


def image2list(path, ratio=0.8):
    dataset = os.path.split(path)[-1].lower()
    print(dataset)
    if dataset not in ("ucf-11","ucf-101"):
        return
    label_file = open("../data/" + dataset + "_labels.txt", 'w')
    train_file = open("../data/" + dataset + "_train_list.txt", 'w')
    test_file = open("../data/" + dataset + "_test_list.txt", 'w')
    if dataset == "ucf-11":
        cnt = 0
        for category in get_dir_list(path):
            label_file.write(os.path.split(category)[-1] + '\t' + str(cnt) + '\n')
            length = 0
            for videos in get_dir_list(category, "Annotation"):
                length += len(get_dir_list(videos))
            cnt2 = 0
            for videos in get_dir_list(category, "Annotation"):
                for clip in get_dir_list(videos):
                    if cnt2 < length * ratio:
                        train_file.write(clip + '\t' + str(cnt) + '\n')
                    else:
                        test_file.write(clip + '\t' + str(cnt) + '\n')
                    cnt2 += 1
            cnt += 1
    elif dataset =='ucf-101':
        cnt = 0
        for category in get_dir_list(path):
            label_file.write(os.path.split(category)[-1] + '\t' + str(cnt) + '\n')
            length = len(get_dir_list(category))
            cnt2 = 0
            for clip in get_dir_list(category):
                if cnt2 < length * ratio:
                    train_file.write(clip + '\t' + str(cnt) + '\n')
                else:
                    test_file.write(clip + '\t' + str(cnt) + '\n')
                cnt2 += 1
            cnt += 1



# generate images for all videos
def video2image(path, frame=4):
    dataset = os.path.split(path)[-1].lower()
    if dataset not in ("ucf-11",'ucf-101'):
        return
    if dataset == "ucf-11":
        for category in get_dir_list(path):
            for videos in get_dir_list(category, "Annotation"):
                for clip in get_file_list(videos):
                    one_video2image(clip, frame)
    elif dataset=='ucf-101':
        for category in get_dir_list(path):
            for clip in get_file_list(category):
                one_video2image(clip, frame)


def one_video2image(infile, frame=5, crop_size=128):
    folder = os.path.splitext(infile)[0]
    # print folder
    os.system("rm -rf %s" % folder)
    os.system("mkdir %s" % folder)
    str = "ffmpeg -i %s -s %d*%d -vf fps=%d %s" % (infile, crop_size, crop_size, frame, folder) + "/%05d.jpg"
    # print str
    os.system(str)


if __name__ == '__main__':
    infile = "/home/charlie/Desktop/dataset/UCF-11/basketball/v_shooting_01/v_shooting_01_01.avi"
    outfile = "/home/charlie/Desktop/1.jpg"

    # print get_file_list("/home/charlie/Desktop/dataset/UCF-11/volleyball_spiking/v_spiking_14/v_spiking_14_04")
    # one_video2image(infile)
    # video2image("/home/charlie/Desktop/dataset/UCF-11")
    # image2list("/home/charlie/Desktop/dataset/UCF-11")
    # video2image("/home/charlie/Desktop/dataset/UCF-101")
    image2list("/home/charlie/Desktop/dataset/UCF-101")