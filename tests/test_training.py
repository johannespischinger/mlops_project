import logging
import os

import pytest
import torch

from src.models.train_model import CNNModel, train
from tests import _PROJECT_ROOT


@pytest.mark.skipif(not os.path.exists(f'{_PROJECT_ROOT}/data'), reason='Data files not found')
def test_train(caplog):
    caplog.set_level(logging.INFO)
    model = CNNModel()
    dataset = torch.load(f'{_PROJECT_ROOT}/data/processed/test.pt')
    train(model, dataset, torch.nn.NLLLoss(), torch.optim.Adam(model.parameters(), lr=0.001),
          epoch=1, batchsize=64, learning_rate=0.001, test=True)
    assert 'Training finished' in caplog.text, 'Model did not run until the end'


