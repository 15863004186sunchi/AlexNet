import torch
from torch import nn, device
from torchvision import datasets
from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, exists


def scan_path(path):
    """
    1. Scan APP path to see if structure is well-established.
    2. Get all filename of Unlabeled images to 'images' list.

    :param path: Path of the project.
    :return images: A list of all filename of unlabeled images.
    """

    """ 1. """
    if not exists(join(path, 'APP')):
        mkdir(join(path, 'APP'))

    for i in range(10):
        if not exists(join(path, 'APP/n'+str(i))):
            mkdir(join(path, 'APP/n'+str(i)))

    """ 2. """
    images = [
        f for f in listdir(path)
        if isfile(join(path, f))]

    return images


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
            ('s1', nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            )),
        ]))

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                padding=2,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
            ('s2', nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            )),
        ]))

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(
                in_channels=256,
                out_channels=384,
                padding=1,
                kernel_size=3,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
        ]))

        self.c4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                padding=1,
            )),
            ('ReLU', nn.ReLU(inplace=True))
        ]))

        self.c5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
            ('s5', nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            )),
        ]))

        self.fc1 = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(
                in_features=6 * 6 * 256,
                out_features=4096,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
            ('drop', nn.Dropout(0.5)),
        ]))

        self.fc2 = nn.Sequential(OrderedDict([
            ('l2', nn.Linear(
                in_features=4096,
                out_features=4096,
            )),
            ('ReLU', nn.ReLU(inplace=True)),
            ('drop', nn.Dropout(0.5)),
        ]))

        self.fc3 = nn.Sequential(OrderedDict([
            ('l3', nn.Linear(
                in_features=4096,
                out_features=10,
            )),
        ]))

    def forward(self, x):
        x = self.c5(self.c4(self.c3(self.c2(self.c1(x)))))
        x = x.view(-1, 6 * 6 * 256)
        x = self.fc3(self.fc2(self.fc1(x)))

        return x


def main():
    path = './'
    images = scan_path(path)


if __name__ == '__main__':
    main()
