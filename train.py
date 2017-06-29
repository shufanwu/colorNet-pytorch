import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from myimgfolder import TrainImageFolder
from colornet import ColorNet

original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

data_dir = "../places205"
train_set = TrainImageFolder(os.path.join(data_dir, 'train'), original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

color_model = ColorNet()
optimizer = optim.Adadelta(color_model.parameters())


def train(epoch):
    color_model.train()
    try:
        messagefile = open('message.txt', 'w')
        for batch_idx, (data, classes) in enumerate(train_loader):
            original_img = data[0].unsqueeze(1).float()
            scale_img = data[1].unsqueeze(1).float()
            lab_img = data[2].float()
            img_ab = lab_img[:, 1:3, :, :]
            original_img, scale_img = Variable(original_img), Variable(scale_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            class_output, output = color_model(original_img, scale_img)
            ems_loss = torch.pow((img_ab-output), 2).sum()/torch.from_numpy(np.array(list(output.size()))).prod()
            print(ems_loss)
            print(F.cross_entropy(class_output, classes))
            loss = ems_loss + 1/300 * F.cross_entropy(class_output, classes)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0])
                messagefile.write(message)
                # print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        messagefile.close()
        torch.save(color_model.state_dict(), 'colornet_params.pkl')


train(20)
