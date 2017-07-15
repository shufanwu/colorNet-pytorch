import os.path
import shutil

import skimage.io as io
import numpy as np
from skimage.color import rgb2lab

rootdir = '../images256/'  # 指明被遍历的文件夹
newrootdir = '../datadelete/'

for parent, dirnames, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    # for dirname in dirnames:                            #输出文件夹信息
    #     print("parent is:" + parent)
    #     print("dirname is:" + dirname)
    messagefile = open('./deletemessage.txt', 'a')
    messagefile.write(os.path.split(parent)[1])
    messagefile.close()

    for filename in filenames:  # 输出文件信息
        # print("parent is:" + parent)
        # print("filename is:" + filename)
        # print("the full name of the file is:" + os.path.join(parent,filename)) #输出文件路径信息
        path = os.path.join(parent, filename)
        img = io.imread(path)

        # remove gray image of 1 channel
        if len(img.shape) == 2:
            newpath = os.path.join(newrootdir, os.path.split(parent)[1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            shutil.move(path, newpath)

        else:
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            num = r.size
            num1 = img.size

            # remove gray image of 3 channel
            if np.sum(abs(r - g) < 30) / num > 0.9 and np.sum(abs(r - b) < 30) / num > 0.9:
                newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                shutil.move(path, newpath)

            else:
                # remove image with small color variance
                img = rgb2lab(img)
                ab = [img[:, :, 1], img[:, :, 2]]
                variance = np.sqrt(np.sum(np.power(ab[0] - np.sum(ab[0]) / num, 2)) / num) + \
                           np.sqrt(np.sum(np.power(ab[1] - np.sum(ab[1]) / num, 2)) / num)
                if variance < 6:
                    newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    shutil.move(path, newpath)
