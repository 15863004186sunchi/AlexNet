import torch
from torch import nn, device
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

GPU = True

DOWNLOAD = True
SHUFFLE = True
NUM_WORKERS = 16
BATCH_SIZE = 16

LR = 0.001
EPOCH = 10

""" 
######################
# Whether to use GPU #
######################
"""
device = device(
    "cuda:0" if (torch.cuda.is_available() and GPU)
    else "cpu")
# Assume that we are on a CUDA machine,
# then this should print a CUDA device:
print(3 * '*',
      ' This calculation takes place in {}'.format(device),
      3 * '*', '\n')

"""
###############
# Data setups #
###############
"""
train_set = datasets.MNIST(
    root='./data',
    train=True,
    download=DOWNLOAD,
    transform=transforms.ToTensor())
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS)

test_set = datasets.MNIST(
    root='./data',
    train=False,
    download=DOWNLOAD,
    transform=transforms.ToTensor())
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS)

"""
#########
# Model #
#########
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        """ (1, 28, 28) -> (6, 28, 28) -> (6, 14, 14) """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                # height=1 in gray
                out_channels=6,
                # 6 filter
                kernel_size=5,
                # filter size
                stride=1,
                padding=2,
                # (28, 28) -> (32, 32)
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                # (28, 28) -> (14, 14)
            ),
        )

        """ (6, 14, 14) -> (36, 10, 10) -> (36, 5, 5) """
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                # height=1 in gray * 6
                out_channels=6 * 6,
                # 6 filter * 6
                kernel_size=5,
                # filter size
                stride=1,
                padding=0,
                # (14, 14)
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                # (10, 10) -> (5, 5)
            ),
        )

        """ Full connection """
        self.out = nn.Sequential(
            nn.Linear(36 * 5 * 5, 270),
            nn.Linear(270, 120),
            nn.Linear(120, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # (batch, 1, 28, 28)
        x = self.conv1(x)
        # (batch, 6, 14, 14)
        x = self.conv2(x)
        # (batch, 36, 5, 5)
        x = x.view(x.size(0), -1)
        # (batch, 36*5*5)
        out_put = self.out(x)
        # (batch, 10)
        return out_put


cnn = CNN().to(device)
# Print net architect
print('The net architecture: \n', cnn, '\n')

"""
###############################
# Optimizer and Loss Function #
###############################
"""
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

"""
####################
# Training process #
####################
"""
print('Start Training: ')
for epoch in range(EPOCH):

    in_data_loss = 0.0
    # initial the in_data_loss var
    for i, data in enumerate(train_loader):

        """ get images and label from data in train loader """
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        """ put data into net """
        output = cnn(images)

        """ calculate loss and optimize parameters """
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """ Print in_data_loss for each small batch """
        small_batch = 250
        # here, the in-data-loss is the average loss of
        # the last small batch. namely, the last 250 data.
        in_data_loss += loss.item()
        # accumulate loss

        if i % small_batch == small_batch - 1:
            print('[Epoch: {:2d}, Data: {:5d}] in data loss: {:.5f}'.format(
                epoch + 1, (i + 1) * BATCH_SIZE,
                in_data_loss / small_batch
            ))
            in_data_loss = 0.0

""" Training Finished """
print('Finish Training\n')

"""
#################
# Out data test #
#################
"""
accuracy = 0.0
correct = 0
total = 0
# initial vars

print('Out-data-test: ')
for i, data in enumerate(test_loader):

    """ get images and label from data in train loader """
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    """ put data into net """
    output = cnn(images)

    """ get prediction """
    # torch.max(input, dim, max=None, max_indices=None)
    # -> (Tensor, LongTensor)
    # max[0] is the abs max in input,
    # max[1] is the index of the abs max in input
    prediction_prob, prediction = torch.max(output, 1)

    """ show label and prediction for some batch """
    # be aware that label and prediction "MUST" be
    # transferred back to CPU if using CUDA so as to
    # transform the tensor into numpy array.
    labels = labels.cpu().numpy()
    prediction = prediction.cpu().numpy()

    if i * BATCH_SIZE % int(10000 / 5) == 0:
        print('true label: {}'.format(labels))
        print('prediction: {}\n'.format(prediction))

    """ calculate total accuracy """
    total += len(labels)
    for j, l in enumerate(labels):
        if l == prediction[j]:
            correct += 1
print(
    3 * '*',
    'Accuracy of the whole database:',
    3 * '*', '\n',

    'Correct: {:d}\n'
    'Total: {:d}\n'
    'Total accuracy: {:.3f}'.format(
        correct,
        total,
        100 * correct / total
    ))

"""
Detail: 
    This sctipt has been tested under Ubuntu 18.04,
    with pytorch 0.4,
    and cuda9.1(without any patch) and cudnn7.1.2,

    About the installation:
    https://mark-down-now.blogspot.com/2018/05/pytorch-gpu-ubuntu-1804.html


Result:

    ***  This calculation takes place in cuda:0 *** 

    The net architecture: 
     CNN(
      (conv1): Sequential(
        (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(6, 36, kernel_size=(5, 5), stride=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (out): Sequential(
        (0): Linear(in_features=900, out_features=270, bias=True)
        (1): Linear(in_features=270, out_features=120, bias=True)
        (2): Linear(in_features=120, out_features=64, bias=True)
        (3): Linear(in_features=64, out_features=10, bias=True)
      )
    ) 

    Start Training: 
    [Epoch:  1, Data:  4000] in data loss: 2.30399
    [Epoch:  1, Data:  8000] in data loss: 2.30210
    [Epoch:  1, Data: 12000] in data loss: 2.30107
    [Epoch:  1, Data: 16000] in data loss: 2.29998
    [Epoch:  1, Data: 20000] in data loss: 2.29815
    [Epoch:  1, Data: 24000] in data loss: 2.29636
    [Epoch:  1, Data: 28000] in data loss: 2.29516
    [Epoch:  1, Data: 32000] in data loss: 2.29406
    [Epoch:  1, Data: 36000] in data loss: 2.29365
    [Epoch:  1, Data: 40000] in data loss: 2.29022
    [Epoch:  1, Data: 44000] in data loss: 2.28779
    [Epoch:  1, Data: 48000] in data loss: 2.28563
    [Epoch:  1, Data: 52000] in data loss: 2.28171
    [Epoch:  1, Data: 56000] in data loss: 2.27827
    [Epoch:  1, Data: 60000] in data loss: 2.27359
    [Epoch:  2, Data:  4000] in data loss: 2.26873
    [Epoch:  2, Data:  8000] in data loss: 2.26362
    [Epoch:  2, Data: 12000] in data loss: 2.25427
    [Epoch:  2, Data: 16000] in data loss: 2.24356
    [Epoch:  2, Data: 20000] in data loss: 2.23107
    [Epoch:  2, Data: 24000] in data loss: 2.21088
    [Epoch:  2, Data: 28000] in data loss: 2.18483
    [Epoch:  2, Data: 32000] in data loss: 2.14416
    [Epoch:  2, Data: 36000] in data loss: 2.07814
    [Epoch:  2, Data: 40000] in data loss: 1.98525
    [Epoch:  2, Data: 44000] in data loss: 1.83186
    [Epoch:  2, Data: 48000] in data loss: 1.67501
    [Epoch:  2, Data: 52000] in data loss: 1.49357
    [Epoch:  2, Data: 56000] in data loss: 1.35056
    [Epoch:  2, Data: 60000] in data loss: 1.14813
    [Epoch:  3, Data:  4000] in data loss: 0.99778
    [Epoch:  3, Data:  8000] in data loss: 0.91442
    [Epoch:  3, Data: 12000] in data loss: 0.79644
    [Epoch:  3, Data: 16000] in data loss: 0.70601
    [Epoch:  3, Data: 20000] in data loss: 0.64328
    [Epoch:  3, Data: 24000] in data loss: 0.59310
    [Epoch:  3, Data: 28000] in data loss: 0.56841
    [Epoch:  3, Data: 32000] in data loss: 0.55499
    [Epoch:  3, Data: 36000] in data loss: 0.51467
    [Epoch:  3, Data: 40000] in data loss: 0.53382
    [Epoch:  3, Data: 44000] in data loss: 0.48603
    [Epoch:  3, Data: 48000] in data loss: 0.44969
    [Epoch:  3, Data: 52000] in data loss: 0.45940
    [Epoch:  3, Data: 56000] in data loss: 0.42092
    [Epoch:  3, Data: 60000] in data loss: 0.43939
    [Epoch:  4, Data:  4000] in data loss: 0.40494
    [Epoch:  4, Data:  8000] in data loss: 0.38511
    [Epoch:  4, Data: 12000] in data loss: 0.38757
    [Epoch:  4, Data: 16000] in data loss: 0.39174
    [Epoch:  4, Data: 20000] in data loss: 0.39832
    [Epoch:  4, Data: 24000] in data loss: 0.35187
    [Epoch:  4, Data: 28000] in data loss: 0.37824
    [Epoch:  4, Data: 32000] in data loss: 0.36439
    [Epoch:  4, Data: 36000] in data loss: 0.34485
    [Epoch:  4, Data: 40000] in data loss: 0.34397
    [Epoch:  4, Data: 44000] in data loss: 0.32093
    [Epoch:  4, Data: 48000] in data loss: 0.32915
    [Epoch:  4, Data: 52000] in data loss: 0.31255
    [Epoch:  4, Data: 56000] in data loss: 0.30638
    [Epoch:  4, Data: 60000] in data loss: 0.32751
    [Epoch:  5, Data:  4000] in data loss: 0.30081
    [Epoch:  5, Data:  8000] in data loss: 0.31306
    [Epoch:  5, Data: 12000] in data loss: 0.31833
    [Epoch:  5, Data: 16000] in data loss: 0.28319
    [Epoch:  5, Data: 20000] in data loss: 0.29209
    [Epoch:  5, Data: 24000] in data loss: 0.25115
    [Epoch:  5, Data: 28000] in data loss: 0.27631
    [Epoch:  5, Data: 32000] in data loss: 0.29087
    [Epoch:  5, Data: 36000] in data loss: 0.27275
    [Epoch:  5, Data: 40000] in data loss: 0.27249
    [Epoch:  5, Data: 44000] in data loss: 0.27865
    [Epoch:  5, Data: 48000] in data loss: 0.24123
    [Epoch:  5, Data: 52000] in data loss: 0.24563
    [Epoch:  5, Data: 56000] in data loss: 0.24946
    [Epoch:  5, Data: 60000] in data loss: 0.25121
    [Epoch:  6, Data:  4000] in data loss: 0.24674
    [Epoch:  6, Data:  8000] in data loss: 0.22715
    [Epoch:  6, Data: 12000] in data loss: 0.23203
    [Epoch:  6, Data: 16000] in data loss: 0.21861
    [Epoch:  6, Data: 20000] in data loss: 0.24425
    [Epoch:  6, Data: 24000] in data loss: 0.21970
    [Epoch:  6, Data: 28000] in data loss: 0.22707
    [Epoch:  6, Data: 32000] in data loss: 0.23027
    [Epoch:  6, Data: 36000] in data loss: 0.21679
    [Epoch:  6, Data: 40000] in data loss: 0.20821
    [Epoch:  6, Data: 44000] in data loss: 0.21273
    [Epoch:  6, Data: 48000] in data loss: 0.21241
    [Epoch:  6, Data: 52000] in data loss: 0.21010
    [Epoch:  6, Data: 56000] in data loss: 0.20716
    [Epoch:  6, Data: 60000] in data loss: 0.20171
    [Epoch:  7, Data:  4000] in data loss: 0.20345
    [Epoch:  7, Data:  8000] in data loss: 0.18528
    [Epoch:  7, Data: 12000] in data loss: 0.19264
    [Epoch:  7, Data: 16000] in data loss: 0.18087
    [Epoch:  7, Data: 20000] in data loss: 0.18629
    [Epoch:  7, Data: 24000] in data loss: 0.17976
    [Epoch:  7, Data: 28000] in data loss: 0.17537
    [Epoch:  7, Data: 32000] in data loss: 0.18168
    [Epoch:  7, Data: 36000] in data loss: 0.15970
    [Epoch:  7, Data: 40000] in data loss: 0.18684
    [Epoch:  7, Data: 44000] in data loss: 0.18961
    [Epoch:  7, Data: 48000] in data loss: 0.17090
    [Epoch:  7, Data: 52000] in data loss: 0.15521
    [Epoch:  7, Data: 56000] in data loss: 0.17310
    [Epoch:  7, Data: 60000] in data loss: 0.18719
    [Epoch:  8, Data:  4000] in data loss: 0.15822
    [Epoch:  8, Data:  8000] in data loss: 0.16008
    [Epoch:  8, Data: 12000] in data loss: 0.14532
    [Epoch:  8, Data: 16000] in data loss: 0.15358
    [Epoch:  8, Data: 20000] in data loss: 0.16307
    [Epoch:  8, Data: 24000] in data loss: 0.15570
    [Epoch:  8, Data: 28000] in data loss: 0.16319
    [Epoch:  8, Data: 32000] in data loss: 0.15586
    [Epoch:  8, Data: 36000] in data loss: 0.15075
    [Epoch:  8, Data: 40000] in data loss: 0.15077
    [Epoch:  8, Data: 44000] in data loss: 0.14796
    [Epoch:  8, Data: 48000] in data loss: 0.14691
    [Epoch:  8, Data: 52000] in data loss: 0.14112
    [Epoch:  8, Data: 56000] in data loss: 0.14874
    [Epoch:  8, Data: 60000] in data loss: 0.14685
    [Epoch:  9, Data:  4000] in data loss: 0.14129
    [Epoch:  9, Data:  8000] in data loss: 0.14410
    [Epoch:  9, Data: 12000] in data loss: 0.14841
    [Epoch:  9, Data: 16000] in data loss: 0.15712
    [Epoch:  9, Data: 20000] in data loss: 0.13227
    [Epoch:  9, Data: 24000] in data loss: 0.13213
    [Epoch:  9, Data: 28000] in data loss: 0.12425
    [Epoch:  9, Data: 32000] in data loss: 0.12657
    [Epoch:  9, Data: 36000] in data loss: 0.12422
    [Epoch:  9, Data: 40000] in data loss: 0.13752
    [Epoch:  9, Data: 44000] in data loss: 0.12457
    [Epoch:  9, Data: 48000] in data loss: 0.12558
    [Epoch:  9, Data: 52000] in data loss: 0.12779
    [Epoch:  9, Data: 56000] in data loss: 0.13082
    [Epoch:  9, Data: 60000] in data loss: 0.13429
    [Epoch: 10, Data:  4000] in data loss: 0.13316
    [Epoch: 10, Data:  8000] in data loss: 0.14140
    [Epoch: 10, Data: 12000] in data loss: 0.11863
    [Epoch: 10, Data: 16000] in data loss: 0.11041
    [Epoch: 10, Data: 20000] in data loss: 0.12114
    [Epoch: 10, Data: 24000] in data loss: 0.12890
    [Epoch: 10, Data: 28000] in data loss: 0.11866
    [Epoch: 10, Data: 32000] in data loss: 0.11555
    [Epoch: 10, Data: 36000] in data loss: 0.11944
    [Epoch: 10, Data: 40000] in data loss: 0.12737
    [Epoch: 10, Data: 44000] in data loss: 0.11413
    [Epoch: 10, Data: 48000] in data loss: 0.11348
    [Epoch: 10, Data: 52000] in data loss: 0.11100
    [Epoch: 10, Data: 56000] in data loss: 0.11641
    [Epoch: 10, Data: 60000] in data loss: 0.12380
    Finish Training

    Out-data-test: 
    true label: [6 0 4 0 4 6 5 7 9 7 1 8 8 9 1 5]
    prediction: [6 0 4 0 4 6 5 7 9 7 1 8 8 9 1 5]

    true label: [3 6 9 2 4 0 2 3 0 7 7 2 0 0 1 1]
    prediction: [3 6 9 2 4 0 2 3 0 7 7 2 0 0 1 1]

    true label: [2 7 7 0 8 7 3 0 3 1 7 2 0 6 8 5]
    prediction: [2 7 7 0 8 7 3 0 3 1 7 2 0 6 8 5]

    true label: [8 3 0 0 0 9 0 6 0 1 3 4 1 8 3 5]
    prediction: [8 3 0 0 0 9 0 6 0 1 3 4 1 8 3 5]

    true label: [1 3 1 9 1 5 1 4 2 1 2 4 4 1 5 9]
    prediction: [1 3 1 9 1 5 1 4 2 1 7 4 4 1 5 9]

    *** Accuracy of the whole database: *** 
     Correct: 9691
    Total: 10000
    Total accuracy: 96.910

    Process finished with exit code 0
"""