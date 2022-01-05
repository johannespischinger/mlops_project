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


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        """
        Class to facilitate train and evaulation of model

        Methods
        -----
        train(self):
            Training model

            Parameters
            ____
            None

            Return
            ___
            None


        evaluate: Evaluate model with testset



        """
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        model = CNNModel()
        train_set = torch.load(
            "/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/cookiecutter-ds-project/\
            data/processed/train"
        )
        epoch = 5
        batchsize = 64
        trainloader = torch.utils.data.DataLoader(train_set, batchsize, shuffle=True)
        lr = 0.003
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
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

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        checkpoint = torch.load(args.load_model_from)
        model = CNNModel()
        model.load_state_dict(checkpoint["state_dict"])
        # !!!! Important to turn off dropout
        model.eval()
        test_set = torch.load(
            "/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/cookiecutter-ds-project/\
            data/processed/test"
        )
        batchsize = 64
        epoch = 5
        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=batchsize, shuffle=True
        )

        #  Important to forbid gradient calculation
        with torch.no_grad():
            running_loss = 0
            test_acc = []
            for images, labels in testloader:
                images = images.unsqueeze(1)
                ps = torch.exp(model(images.float()))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

            else:
                print(f"Accuracy: {sum(test_acc) / len(test_acc)}")


if __name__ == "__main__":
    TrainOREvaluate()
