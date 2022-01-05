import argparse
import os

import numpy as np

import torch
from src.models.train_model import CNNModel


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

    # Loading the files
    images = []
    labels = []
    path = args.load_data_from
    if os.path.isfile(path) == True:
        prediction = np.load(path)
        images.append((prediction["images"]))
        labels.append((prediction["labels"]))

    # else:
    #    for file in glob(glob(os.path.join(path,"*"))):
    #        .append(np.load(file))

    dataset = torch.from_numpy(np.asarray(images)).view(5000, 1, 28, 28)

    with torch.no_grad():
        prediction = torch.exp(model(dataset.float()))
        top_p, top_class = prediction.topk(1, dim=1)

    return model, prediction, top_p, top_class


if __name__ == "__main__":
    model, prediction, _, _ = loader()
