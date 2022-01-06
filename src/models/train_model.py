import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn


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


def train(self, model: nn.Model, epoch: float = 5, batchsize: int = 64, dataset: torch.nn.Dataset, learning_rate: float = 0.001,\
          loss: torch.nn.Loss,\
          optimizer: torch.optim):

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True)

    criterion = loss
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

    modelstring = str(f"b_{batchsize}_e_{epoch}_lr_{lr}")
    torch.save(
        checkpoint,
        f"/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/cookiecutter-ds-project/\
        models/trained_model_{modelstring}.pt",
    )

    plt.plot(loss_hist)
    plt.savefig(
        f"/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/cookiecutter-ds-project/\
        reports/figures/training_curve_{modelstring}.png"
    )

if __name__ == "__main__":
    model = CNNModel()
    dataset_path = 'data/processed/train'
    train(model,epoch = 5, batchsize= 64, dataset_path, learning_rate= 0.001, loss = torch.nn.NLLLoss(), optimizer= torch.optim.Adam(model.parameters(),learning_rate = 0.001))

