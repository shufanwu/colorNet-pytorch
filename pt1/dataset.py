import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab, rgb2gray

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(256),
    transforms.Resize(224)
])


class ColorDataset(Dataset):
    def __init__(self, phase):
        assert (phase in ['train', 'val', 'test'])
        self.phase = phase
        self.root_dir = '/home/wsf/Pictures/Wallpapers'
        self.samples = None
        with open('{}/labels/{}.txt'.format(self.root_dir, phase), 'r') as f:
            self.samples = f.readlines()[:3]

        print('[+] dataset `{}` loaded {} images'.format(self.phase, len(self.samples)))

    def __getitem__(self, idx):
        if self.phase == 'train' or self.phase == 'val':
            image_path, label = self.samples[idx].strip().split()
            label = np.array(int(label))
            image = Image.open('{}/images/{}'.format(self.root_dir, image_path)).convert('RGB')
            image = data_augmentation(image)
            image = np.asarray(image)
            img_lab = rgb2lab(image)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3].astype('float32')
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_gray = rgb2gray(image).astype('float32')
            img_gray = img_gray[np.newaxis, :]
            img_gray = torch.from_numpy(img_gray)
            return (img_gray, img_ab), label

        else:
            image_path = self.samples[idx].strip()
            img_gray = Image.open('{}/images/{}'.format(self.root_dir, image_path)).convert('L')
            img_gray_scale = img_gray.copy()
            img_gray_scale = img_gray_scale.resize((224, 224))
            img_gray = transforms.ToTensor()(img_gray)
            img_gray_scale = transforms.ToTensor()(img_gray_scale)
            return img_gray, img_gray_scale

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    pass
