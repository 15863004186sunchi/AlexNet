import torch
from torch import nn, device
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

hardware = {
    'GPU': True,
}
data_sets_settings = {
    'ROOT': './data',
    'DOWNLOAD': True,
    'SHUFFLE': True,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 16,
}
hyper = {
    'LR': 1e-2,
    'EPOCH': 5,
    'BATCH_SIZE': 16,
    'SMALL_BATCH': 250,
}

def device_(settings):
    """
    Assume that we are on a CUDA machine,
    then this should print a CUDA device;
    else should print 'cpu'.

    :param settings: dict of devices settings.
    :return devices: devices that the net will take place.
    """

    devices = device(
        "cuda:0" if (torch.cuda.is_available() and settings['GPU'])
        else "cpu"
    )

    print(
        '\n', 3 * '*',
        ' This calculation takes place in {}'.format(devices),
        3 * '*', '\n'
    )

    return devices

def data(settings):
    """
    data sets.

    :param settings: dict of data sets settings.
    :return train_loader: train data loader.
    :return test_loader: test data loader.
    """

    train_set = datasets.MNIST(
        root=settings['ROOT'],
        train=True,
        download=settings['DOWNLOAD'],
        transform=transforms.ToTensor()
    )

    test_set = datasets.MNIST(
        root=settings['ROOT'],
        train=False,
        download=settings['DOWNLOAD'],
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_set,
        batch_size=settings['BATCH_SIZE'],
        shuffle=settings['SHUFFLE'],
        num_workers=settings['NUM_WORKERS']
    )

    test_loader = DataLoader(
        test_set,
        batch_size=settings['BATCH_SIZE'],
        shuffle=settings['SHUFFLE'],
        num_workers=settings['NUM_WORKERS']
    )

    return train_loader, test_loader

class CNN(nn.Module):
    """
    Input:
        batch*01*28*28

    ###############
    # Convolution #
    ###############
    C1:
        batch*01*28*28 (padding)->
        batch*01*32*32 (convolution)->
        batch*16*28*28
    ReLu
    pool1:
        batch*16*28*28 (max pooling)->
        batch*16*14*14
    C2:
        batch*16*14*14 (padding)->
        batch*36*16*16 (convolution)->
        batch*36*12*12
    ReLu
    pool2:
        batch*36*12*12 (max pooling)->
        batch*36*6*6
    drop out: 0.25

    #####################
    # Linear Connection #
    #####################
    f3:
        batch*36*6*6 (Linear)->
        batch*128
    ReLu
    drop out: 0.25
    f4:
        batch*128 (Linear)->
        batch*10
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.convolution_net = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(
                 in_channels=1,
                 out_channels=16,
                 kernel_size=5,
                 stride=1,
                 padding=2,
             )),
            ('relu', nn.ReLU()),
            ('pool1', nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )),
            ('c2', nn.Conv2d(
                in_channels=16,
                out_channels=36,
                kernel_size=5,
                padding=1,
            )),
            ('relu', nn.ReLU()),
            ('pool2', nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )),
            ('dropout', nn.Dropout2d(0.25)),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f3', nn.Linear(
                in_features=36*6*6,
                out_features=128,
            )),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(0.25)),
            ('f4', nn.Linear(
                in_features=128,
                out_features=10,
            )),
        ]))

    def forward(self, x):
        x = self.convolution_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(devices, training_data, net, hyper_p, testing_data=(False, None)):
    """

    :param devices:
    :param training_data:
    :param net:
    :param hyper_p:
    :param testing_data:
    :return:
    """

    """ Optimizer and Loss Function """
    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_p['LR'])
    loss_func = nn.CrossEntropyLoss()

    print('The net architecture: \n', net, '\n')

    print('Start Training: ')
    for epoch in range(hyper_p['EPOCH']):

        """ initial the in_data_loss var """
        in_data_loss = 0.0
        for i, _data in enumerate(training_data):

            """ get images and label from data in train loader """
            images, labels = _data
            images, labels = images.to(devices), labels.to(devices)

            """ put data into net """
            output = net(images)

            """ calculate loss and optimize parameters """
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ Print in_data_loss for each small batch """
            small_batch = hyper_p['SMALL_BATCH']
            # here, the in-data-loss is the average loss of
            # the last small batch. namely, the last 250 data.
            in_data_loss += loss.item()
            # accumulate loss

            if i % small_batch == small_batch - 1:
                print('[Epoch: {:2d}, Data: {:5d}] in data loss: {:.5f}'.format(
                    epoch + 1, (i + 1) * hyper_p['BATCH_SIZE'],
                    in_data_loss / small_batch
                ))
                in_data_loss = 0.0

        if testing_data[0]:
            out_data_test(devices, testing_data[1], net)

def out_data_test(devices, testing_data, net):
    """

    :param devices:
    :param testing_data:
    :param net:
    :return:
    """

    """ initial vars """
    correct = 0
    total = 0
    print('Out-data test: ')
    for i, _t_data in enumerate(testing_data):
        """ get images and label from data in train loader """
        t_images, t_labels = _t_data
        t_images, t_labels = t_images.to(devices), t_labels.to(devices)

        """ put data into net """
        t_output = net(t_images)

        """ get prediction """
        # torch.max(input, dim, max=None, max_indices=None)
        # -> (Tensor, LongTensor)
        # max[0] is the abs max in input,
        # max[1] is the index of the abs max in input
        prediction_prob, prediction = torch.max(t_output, 1)

        """ show label and prediction for some batch """
        # be aware that label and prediction "MUST" be
        # transferred back to CPU if using CUDA so as to
        # transform the tensor into numpy array.
        t_labels = t_labels.cpu().numpy()
        prediction = prediction.cpu().numpy()

        """ calculate total accuracy """
        total += len(t_labels)
        for j, l in enumerate(t_labels):
            if l == prediction[j]:
                correct += 1
    print(
        'Correct: {:d}\n'
        'Total: {:d}\n'
        'Total accuracy: {:.2f}\n'.format(
            correct,
            total,
            100 * correct / total
        ))

def main():
    devices = device_(hardware)
    training_data, testing_data = data(data_sets_settings)
    cnn = CNN().to(devices)
    train(devices, training_data, cnn, hyper, (True, testing_data))

if __name__ == '__main__':
    main()
