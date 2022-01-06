import argparse
import sys
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, List


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
            # Size of image 7*7
        )
        # n_channels = self.convolution(torch.empty(1, 8, 7, 7)).size(-1)
        self.linear = nn.Sequential(
            nn.Linear(392, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):

        x = self.convolution(x)
        x = self.linear(x)

        return x


def train(model: nn.Module,
          dataset,
          criterion: Union[Callable, nn.Module],
          optimizer: Optional[torch.optim.Optimizer],
          epoch: float = 5,
          batchsize: int = 64,
          learning_rate: float = 0.001
          ):

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True)
    loss_hist = []
    for e in range(epoch):
        running_loss = 0
        acc = []
        for images, labels in trainloader:
            optimizer.zero_grad()

            images = images.unsqueeze(1)

            output = model(images.float())
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc.append(torch.mean(equals.type(torch.FloatTensor)))

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        else:
            print(f"Training loss: {running_loss / len(trainloader)}")
            print(f"Accuracy: {sum(acc) / len(acc)}")

        loss_hist.append(running_loss)

    checkpoint = {
        "input_size": (64, 1, 28, 28),
       "output_size": 10,
        "optimizer_state_dicts": optimizer.state_dict(),
        "state_dict": model.state_dict(),
    }

    modelstring = str(f"b_{batchsize}_e_{epoch}_lr_{learning_rate}")
    os.makedirs("models/", exist_ok=True)
    torch.save(
        checkpoint,
        "models/trained_model.pt")


    #plt.plot(loss_hist)
    #plt.savefig(
    #    f"../../reports/figures/training_curve_{modelstring}.png"
    #)

if __name__ == "__main__":
    model = CNNModel()
    dataset = torch.load('../../data/processed/test')

    train(model, dataset, torch.nn.NLLLoss(), torch.optim.Adam(model.parameters(), lr = 0.001),
          epoch = 5, batchsize = 64, learning_rate = 0.001)

