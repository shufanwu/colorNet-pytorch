import os

import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from skimage import io
from colornet import ColorNet
from myimgfolder import ValImageFolder

data_dir = "../places205"

val_set = ValImageFolder(os.path.join(data_dir, 'val'))
val_set_size = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

color_model = ColorNet()
color_model.load_state_dict(torch.load('color_net_params.pkl'))


def val():
    color_model.eval()

    i = 0
    for data, _ in val_loader:
        original_img = data[0].unsqueeze(1).float()
        w = original_img.size()[2]
        h = original_img.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)
        _, output = color_model(original_img, scale_img)
        color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.squeeze().numpy().transpose((1, 2, 0))
        color_img = lab2rgb(color_img)
        color_name = './'+str(i)+'.jpg'
        i += 1
        io.imsave(color_name, color_img)

val()