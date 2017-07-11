import os.path

import shutil
import skimage.io as io
import numpy as np

rootdir = '../data/vision/torralba/deeplearning/images256/'               # 指明被遍历的文件夹
newrootdir = '../datadelete/'

for parent, dirnames, filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    # for dirname in dirnames:                            #输出文件夹信息
    #     print("parent is:" + parent)
    #     print("dirname is:" + dirname)

    for filename in filenames:                          #输出文件信息
        # print("parent is:" + parent)
        # print("filename is:" + filename)
        # print("the full name of the file is:" + os.path.join(parent,filename)) #输出文件路径信息
        path = os.path.join(parent, filename)
        img = io.imread(path)
        if len(img.shape) == 2:
            newpath = os.path.join(newrootdir, os.path.split(parent)[1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if not os.path.exists(os.path.join(newpath, filename)):
                shutil.move(path, newpath)
        else:
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            img = img.transpose((2,0,1))
            num = r.size
            if np.sum(abs(r - g) < 30) / num > 0.9 and np.sum(abs(r - b) < 30) / num > 0.9:
                newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                if not os.path.exists(os.path.join(newpath, filename)):
                    shutil.move(path, newpath)
            else:
                variance = np.sum(np.sqrt(np.sum(np.power(channel - np.sum(channel)/num, 2))/num) for channel in img)
                if variance < 130:
                    newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    if not os.path.exists(os.path.join(newpath, filename)):
                        shutil.move(path, newpath)