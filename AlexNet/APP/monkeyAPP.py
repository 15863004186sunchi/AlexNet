from collections import OrderedDict
from os import listdir, mkdir, rename
from os.path import isfile, join, exists

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor


def scan_path(path):
    """
    1. Scan APP path to see if structure is well-established.
    2. Get all filename of Unlabeled images to 'images' list.

    :param path: Path of the project.
    :return images: A list of all filename of unlabeled images.
    """

    """ 1. """
    if not exists(path):
        mkdir(path)

    for i in range(10):
        if not exists(join(path, 'n' + str(i))):
            mkdir(join(path, 'n' + str(i)))

    """ 2. """
    images = [
        [join(path, f), f] for f in listdir(path)
        if isfile(join(path, f))]

    return images


def image_loader(images):
    transform = Compose([
        Resize(size=(227, 227)),
        ToTensor(),
    ])

    image_set = []
    for image_name in images:
        image = Image.open(image_name[0])
        image = transform(image)

        """ 
        Due to image is a single image batch,
        namely, it contents only one image in the batch.
        Yet, torch.nn only accept 4-D tensor -- batch-like tensor.
        Therefore, it is necessary to add a dim with value '1'
        as the batch size.
        
        As for this case, 
        (3x227x227) -> (1x3x227x227)
        Just add 'print(image.shape)' before and after 'UnSqueeze'.
        """
        image = image.unsqueeze(0)
        # image = image.cuda()
        image_set.append(image)

    return image_set


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
    monkeys = {
        0: 'Alouatta Palliata',
        1: 'Erythrocebus Patas',
        2: 'Cacajao Calvus',
        3: 'Macaca Fuscata',
        4: 'Cebuella Pygmea',
        5: 'Cebus Capucinus',
        6: 'Mico Argentatus',
        7: 'Saimiri Sciureus',
        8: 'Aotus Nigriceps',
        9: 'Trachypithecus Johnii',
    }

    path = './classes'
    original_images = scan_path(path)
    images = image_loader(original_images)
    net = AlexNet()
    net.load_state_dict(torch.load('alex_net.pkl'))
    for i, im in enumerate(images):
        output = net(im)
        prediction_prob, prediction = torch.max(output, 1)
        print(prediction_prob)
        print(
            'file: "{}" is recognized as type {}\n'.format(
                original_images[i][1], prediction.numpy()[0]),
            'and labeled as {}\n'.format(
                monkeys[prediction.numpy()[0]]))

        """ move image to class """
        rename(
            original_images[i][0],
            join(
                join(path, 'n' + str(prediction.numpy()[0])),
                original_images[i][1]))


if __name__ == '__main__':
    main()
