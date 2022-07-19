from __future__ import print_function
from datetime import datetime
import json
import os
import uuid
import numpy as np
from torch.utils.data import TensorDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import Net, BinaryNet

# Training settings
parser = argparse.ArgumentParser(
    description="First binarized DoA estimation example")
parser.add_argument(
    "--bnn", action="store_true", default=False, help="Train BNN instead of DNN (default: False)"
)
parser.add_argument(
    "--generate-data", action="store_true", default=False, help="Generate data (default: False)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for training (default: 512)",
)
parser.add_argument(
    "--data-size",
    type=int,
    default=1000000,
    metavar="N",
    help="data size in samples (default: 1e6)",
)
parser.add_argument(
    "--train-test-ratio",
    type=int,
    default=0.9,
    metavar="N",
    help="percentage used for training (default: 0.9)",
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
    help="learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="Adam momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--sources",
    type=int,
    default=1,
    metavar="N",
    help="Number of sources (default: 1)",
)
parser.add_argument(
    "--realizations",
    type=int,
    default=1,
    metavar="N",
    help="Number of realizations (default: 1)",
)
parser.add_argument(
    "--array-elements",
    type=int,
    default=1024,
    metavar="N",
    help="Array elements (default: 1024)",
)
parser.add_argument(
    "--snr",
    type=int,
    default=1000,
    metavar="N",
    help="Signal to noise ratio (default: 1000)",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_size = int(args.data_size * args.train_test_ratio)
test_size = int(args.data_size - train_size)

if args.generate_data:
    from data_generation import generate_all_data
    # L, N, SNR, numrealization, data_size
    generate_all_data(args.sources, args.array_elements,
                      args.snr, args.realizations, args.data_size)

with open("signal.npy", "rb") as f:
    data = np.load(f)

with open("label.npy", "rb") as f:
    label = np.load(f)

data = torch.Tensor(data)
label = torch.Tensor(label).to(torch.long)

train_dataset = TensorDataset(
    data[:train_size], label[:train_size])  # create your datset
test_dataset = TensorDataset(
    data[train_size:], label[train_size:])  # create your datset

#kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_size, shuffle=True
)

model = None

if args.bnn:
    model = BinaryNet()
else:
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

        if args.bnn:
            output = model(torch.sign(data))
        else:
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


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def test(epoch, path):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            if args.bnn:
                output = model(torch.sign(data))
            else:
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

    save_checkpoint(model, optimizer,
                    f'{path}/model_{epoch=}', epoch)

    return test_loss, 100.0 * correct / len(test_loader.dataset)


def main():
    name = uuid.uuid4().hex
    now = datetime.now()

    path = f'{now}_{name}'

    os.mkdir(f'{path}')
    with open(f'{path}/history.txt', 'w') as history, open('arguments.json', 'w') as args_json:
        history.write("epoch, loss, accuracy")
        json.dump(args.__dict__, args_json, indent=2)

        for epoch in range(1, args.epochs + 1):
            train(epoch)
            loss, accuracy = test(epoch, path)
            history.write(f"{epoch}, {loss}, {accuracy}")
            # if epoch % 40 == 0:
            #    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1


if __name__ == "__main__":
    main()
