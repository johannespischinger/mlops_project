from src.models.train_model import CNNModel,train
import logging
import torch
import os
from pathlib import Path
from tests import _PATH_DATA


def test_train(caplog):
    caplog.set_level(logging.INFO)
    model = CNNModel()
    dataset = torch.load(f'{_PATH_DATA}/data/processed/test.pt')
    train(model, dataset, torch.nn.NLLLoss(), torch.optim.Adam(model.parameters(), lr=0.001),
          epoch=1, batchsize=64, learning_rate=0.001)
    assert 'Training finished' in caplog.text, 'Model did not run until the end'


