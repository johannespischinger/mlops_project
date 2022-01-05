# -*- coding: utf-8 -*-
import os.path
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    train_images = []
    train_labels = []

    for idx in range(5):
        train_data = np.load(os.path.join(input_filepath, "train_{}.npz".format(idx)))
        train_images.append(train_data["images"])
        train_labels.append(train_data["labels"])

    train_images = torch.from_numpy(np.asarray(train_images))
    train_labels = torch.from_numpy(np.asarray(train_labels))

    # Formating tensor from (5,5000,28,28) to (5*5000,28,28)
    train_images = train_images.view(5 * 5000, 28, 28)

    # Flattening from shape (5*5000,1) to (5*5000,)
    train_labels = train_labels.view(5 * 5000, -1).flatten()
    train = torch.utils.data.TensorDataset(train_images, train_labels)

    test_data = np.load(os.path.join(input_filepath, "test.npz"))
    test_images = test_data["images"]
    test_labels = test_data["labels"]
    test_images = torch.from_numpy(np.asarray(test_images))
    test_labels = torch.from_numpy(np.asarray(test_labels))
    test_images = test_images.view(5000, 28, 28)
    test_labels = test_labels.view(5000, -1).flatten()
    test = torch.utils.data.TensorDataset(test_images, test_labels)

    torch.save(train, os.path.join(output_filepath, "train"))
    torch.save(test, os.path.join(output_filepath, "test"))


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
