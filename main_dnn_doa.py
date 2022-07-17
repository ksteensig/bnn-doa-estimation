from __future__ import print_function
import pdb
import numpy as np
from torch.utils.data import TensorDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(
    description="First binarized DoA estimation example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for training (default: 256)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=100000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--gpus", default=1,
                    help="gpus used for training - e.g 0,1,3")
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


with open("signal.npy", "rb") as f:
    data = np.load(f)

with open("label.npy", "rb") as f:
    label = np.load(f)

tensor_x = torch.Tensor(data)  # transform to torch tensor
tensor_y = torch.Tensor(label).to(torch.long)

train_dataset = TensorDataset(
    tensor_x[:900000], tensor_y[:900000])  # create your datset
test_dataset = TensorDataset(
    tensor_x[900000:], tensor_y[900000:])  # create your datset

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=True
)


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


model = Net()
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch % 40 == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p, "org"):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, "org"):
                p.org.copy_(p.data.clamp_(-1, 1))

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    if epoch % 40 == 0:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
