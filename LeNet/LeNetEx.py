from os.path import abspath, dirname
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as p

# hyper parameters
EPOCH = 1000
BATCH_SIZE = 500
LR = 0.001
DOWNLOAD_MNIST = True

minst_data = torchvision.datasets.MNIST(
    root=dirname(abspath(__file__))+'/data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# data vision
def show_data(data, label, i):
    p.imshow(
        data[i].numpy(),
        cmap='gray'
    )
    p.title('{}'.format(label[i]))
    p.show()

train_loader = Data.DataLoader(
    dataset=minst_data,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=8
)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=60,
                kernel_size=5,
                stride= 1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=60,
                out_channels=160,
                kernel_size=5,
                stride= 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            ),
        )
        self.out = nn.Linear(
            in_features=160*5*5,
            out_features=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), 160*5*5)
        output = self.out(x)
        return output

cnn = LeNet()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

i = 0
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        if step % 100 == 0:
            print("Iteration: {}. Loss: {}.".format(i, loss.item()))