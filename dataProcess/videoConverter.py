import os


def get_dir_list(path, _except=None):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and x != _except]


def get_file_list(path, _except=None):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and x != _except]

def image2list():
    pass

# generate images for all videos
def video2image(path, frame=5):
    dataset = os.path.split(path)[-1]
    if dataset not in ("UCF-11"):
        return
    if dataset == "UCF-11":
        for folder in get_dir_list(path):
            print folder
            for subfolder in get_dir_list(folder, "Annotation"):
                for file in get_file_list(subfolder):
                    one_video2image(file, frame)
            break


def one_video2image(infile, frame=5):
    folder = os.path.splitext(infile)[0]
    # print folder
    os.system("rm -rf %s" % folder)
    os.system("mkdir %s" % folder)
    str = "ffmpeg -i %s -vf fps=%d %s" % (infile, frame, folder) + "/%05d.jpg"
    # print str
    os.system(str)


if __name__ == '__main__':
    infile = "/home/charlie/Desktop/dataset/UCF-11/basketball/v_shooting_01/v_shooting_01_01.avi"
    outfile = "/home/charlie/Desktop/1.jpg"
    # one_video2image(infile)
    video2image("/home/charlie/Desktop/dataset/UCF-11")
