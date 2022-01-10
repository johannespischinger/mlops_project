import pytest
from src.models.train_model import CNNModel
import torch

@pytest.mark.parametrize('inputs,output', [([1,1,28,28],392),
                                           ([1,1,32,32],512)])
def test_input(inputs, output):
    sample = torch.randint(0,255,size=inputs)

    model = CNNModel(input_shape=inputs)

    assert len(model.forward_cnn(sample.float()).flatten()) == output