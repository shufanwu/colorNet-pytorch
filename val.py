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


data_dir = "../places205"
have_cuda = torch.cuda.is_available()

val_set = ValImageFolder(data_dir)
val_set_size = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

color_model = ColorNet()
color_model.load_state_dict(torch.load('colornet_params.pkl'))
if have_cuda:
    color_model.cuda()


def val():
    color_model.eval()

    i = 0
    for data, _ in val_loader:
        original_img = data[0].unsqueeze(1).float()
        gray_name = './gray/' + str(i) + '.jpg'
        for img in original_img:
            pic = img.squeeze().numpy()
            pic = pic.astype(np.float64)
            plt.imsave(gray_name, pic, cmap='gray')
        w = original_img.size()[2]
        h = original_img.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        if have_cuda:
            original_img, scale_img = original_img.cuda(), scale_img.cuda()

        original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)
        _, output = color_model(original_img, scale_img)
        color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            color_name = './colorimg/' + str(i) + '.jpg'
            plt.imsave(color_name, img)
            i += 1
        # use the follow method can't get the right image but I don't know why
        # color_img = torch.from_numpy(color_img.transpose((0, 3, 1, 2)))
        # sprite_img = make_grid(color_img)
        # color_name = './colorimg/'+str(i)+'.jpg'
        # save_image(sprite_img, color_name)
        # i += 1

val()