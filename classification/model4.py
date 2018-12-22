import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

# This class is initialized using:
# model = Net(models.resnet50(pretrained=True))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.conv6_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4608, 2000)
        self.fc1_bn = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, nclasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x):
        x = self.conv1_bn(F.leaky_relu(self.conv1(x)))
        x = self.conv2_bn(F.leaky_relu(self.conv2(x)))
        x = self.conv3_bn(F.leaky_relu(self.conv3(x)))
        x = self.conv4_bn(F.leaky_relu(F.max_pool2d(self.conv4(x), 2)))
        x = self.conv5_bn(F.leaky_relu(F.max_pool2d(self.conv5(x), 2)))
        x = self.conv6_bn(F.leaky_relu(self.conv6_drop(self.conv6(x))))
        x = x.view(-1, 4608)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)