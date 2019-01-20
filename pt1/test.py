import os

import torch
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from skimage.color import lab2rgb
from skimage import io
from colornet import ColorNet
from myimgfolder import ValImageFolder
import numpy as np
import matplotlib.pyplot as plt
from colornet import ColorNet
from pt1.dataset import ColorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
BZ=1
test_set = ColorDataset('test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BZ, shuffle=False, num_workers=4)
color_model = ColorNet()
color_model.load_state_dict(torch.load('/home/wsf/colornet_params.pkl'))
color_model.to(device)


def test():
    color_model.eval()

    for idx, (imgs, imgs_scale) in enumerate(test_loader):
        imgs = imgs.to(device)
        imgs_scale = imgs_scale.to(device)
        gray_name = test_set.samples[idx].strip().split('/')[-1]
        for img in imgs:
            pic = img.cpu().squeeze().numpy()
            pic = pic.astype(np.float64)
            plt.imsave('./{}/{}'.format('grayimg',gray_name), pic, cmap='gray')
        w = imgs.size(2)
        h = imgs.size(3)

        _, output = color_model(imgs, imgs_scale)
        color_img = torch.cat((imgs, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            color_name = './colorimg/{}'.format(gray_name)
            plt.imsave(color_name, img)
        # use the follow method can't get the right image but I don't know why
        # color_img = torch.from_numpy(color_img.transpose((0, 3, 1, 2)))
        # sprite_img = make_grid(color_img)
        # color_name = './colorimg/'+str(i)+'.jpg'
        # save_image(sprite_img, color_name)
        # i += 1

if __name__ == '__main__':
    test()
