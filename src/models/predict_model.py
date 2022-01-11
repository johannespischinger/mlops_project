import argparse
import os

import numpy as np
import torch
from train_model import CNNModel


def loader():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("load_model_from", default="")
    parser.add_argument("load_data_from", default="")
    # add any additional argument that you want
    args = parser.parse_args()

    # Loading the model
    checkpoint = torch.load(args.load_model_from)
    model = CNNModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    path = args.load_data_from
    dataset = torch.load(path)

    return model, dataset

def prediction(model,data):

    testloader = torch.utils.data.DataLoader(data)
    test_acc = []

    with torch.no_grad():
        test_acc = []
        for images, labels in testloader:
            prediction = torch.exp(model(images.float().unsqueeze(1)))
            top_p, top_class = prediction.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_acc.append(torch.mean(equals.type(torch.FloatTensor)))

        else:
            print(f"Accuracy: {sum(test_acc) / len(test_acc)}")


if __name__ == "__main__":
    model, dataset = loader()
    prediction(model, dataset)
