import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn

import wandb

Logger = logging.getLogger(__name__)

class CNNModel(nn.Module):
    def __init__(self, input_shape=[1,1,28,28]):
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
        n_channels = self.convolution(torch.empty(input_shape)).size(-1)
        self.linear = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expect 4D tensor')
        if x.shape[1] != 1 :
            raise ValueError('Expected channel size not equal to 1')

        x = self.convolution(x)
        x = self.linear(x)

        return x

    def forward_cnn(self,x):
        if x.ndim != 4:
            raise ValueError('Expect 4D tensor')
        if x.shape[1] != 1 :
            raise ValueError('Expected channel size not equal to 1')

        x = self.convolution(x)

        return x.flatten()


def train(model: nn.Module,
          dataset,
          criterion: Union[Callable, nn.Module],
          optimizer: Optional[torch.optim.Optimizer],
          epoch: float = 10,
          batchsize: int = 64,
          learning_rate: float = 0.001,
          test = False
          ):
    Logger.info('Starting training...')

    if test == True:
        os.environ['WANDB_SILENT'] = "true"
        wandb.init(mode='disabled')
    elif test == False:
        wandb.init(mode='online')


    wandb.watch(model,log_freq=100)
    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True)
    loss_hist = []
    my_table = wandb.Table(columns=['Image', 'Truth', 'Prediction'])

    for e in range(epoch):
        running_loss = 0
        acc = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(images.float())
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc.append(torch.mean(equals.type(torch.FloatTensor)))

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                [my_table.add_data(wandb.Image(image), label.item(), pred.item()) for image, label, pred in
                 zip(images, labels, top_class)]
                wandb.log({"loss": loss})


        else:
            print(f"Training loss: {running_loss / len(trainloader)}")
            print(f"Accuracy: {sum(acc) / len(acc)}")

        loss_hist.append(running_loss)

    wandb.log({'MNIST_prediction': my_table})

    Logger.info('Training finished')
    #my_table.add_column('Label', labels.view(*top_class.shape)[:5])
    #my_table.add_column('Prediction', top_class[:5])
    # wandb.log({'examples': [wandb.Image(im) for im in images[:5]]})


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

    path = Path(__file__).resolve().parents[2]
    print(path.parent.absolute())

    model = CNNModel()
    dataset = torch.load(os.path.join(path,'data/processed/test.pt'))

    train(model, dataset, torch.nn.NLLLoss(), torch.optim.Adam(model.parameters(), lr = 0.001),
          epoch = 1, batchsize = 64, learning_rate = 0.001)

