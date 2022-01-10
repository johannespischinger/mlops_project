import torch
from pathlib import Path
import os
import numpy as np
import pytest
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/data/processed'), reason='Data files not found')
def test_load():
    test = torch.load(f'{_PATH_DATA}/data/processed/test.pt')
    train = torch.load(f'{_PATH_DATA}/data/processed/train.pt')
    #test, train = load()
    assert len(train) == 25000, 'Dataset did not have the correct number of samples'
    assert len(test) == 5000, 'Dataset did not have the correct number of samples'
    assert [list(image.shape) == [1,28,28] for image,_ in test], 'Shape of sample is not correct'
    assert [list(image.shape) == [1, 28, 28] for image, _ in train], 'Shape of sample is not correct'

    train_labels = []
    test_labels = []
    for _,label in train:
        train_labels.append(label)
    for _, label in test:
        test_labels.append(label)

    assert len(torch.from_numpy(np.asarray(train_labels)).unique()) == 10, 'Not all labels are represented in the training dataset'
    assert len(torch.from_numpy(np.asarray(test_labels)).unique()) == 10, 'Not all labels are represented in the test dataset'


