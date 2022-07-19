import torch.nn as nn
from BinaryNet.models.binarized_modules import BinarizeLinear


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = nn.Linear(2*1024, 2048 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc2 = nn.Linear(
            2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc3 = nn.Linear(
            2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc4 = nn.Linear(2048 * self.infl_ratio, 90)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


class BinaryNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(2*1024, 2048 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc2 = BinarizeLinear(
            2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc3 = BinarizeLinear(
            2048 * self.infl_ratio, 2048 * self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc4 = nn.Linear(2048 * self.infl_ratio, 90)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)
