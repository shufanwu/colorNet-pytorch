import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from pt1.dataset import ColorDataset
import numpy as np
from torch.utils.data import DataLoader
from colornet import ColorNet


device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

train_set = ColorDataset('train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = ColorNet()
if os.path.exists('/home/wsf/colornet_params.pkl'):
    color_model.load_state_dict(torch.load('/home/wsf/colornet_params.pkl'))
color_model.to(device)
optimizer = optim.Adadelta(color_model.parameters())

def train():
    color_model.train()
    try:
        for epoch in range(20):
            for batch_idx, (data, label) in enumerate(train_loader):
                messagefile = open('./message.txt', 'a')
                img_gray,img_ab = data
                img_gray = img_gray.to(device)
                img_ab = img_ab.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                class_output, output = color_model(img_gray, img_gray)
                ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
                cross_entropy_loss = 1/300 * F.cross_entropy(class_output, label)
                loss = ems_loss + cross_entropy_loss
                lossmsg = 'loss: %.9f\n' % (loss.item())
                messagefile.write(lossmsg)
                loss.backward()
                # ems_loss.backward(retain_variables=True)
                # cross_entropy_loss.backward()
                optimizer.step()
                if batch_idx % 500 == 0:
                    message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item())
                    messagefile.write(message)
                    torch.save(color_model.state_dict(), 'colornet_params.pkl')
                messagefile.close()
                    # print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                    #     epoch, batch_idx * len(data), len(train_loader.dataset),
                    #     100. * batch_idx / len(train_loader), loss.data[0]))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')


if __name__ == '__main__':
    train()
