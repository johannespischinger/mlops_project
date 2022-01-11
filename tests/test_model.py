import pytest
import torch

from src.models.train_model import CNNModel


def test_model():
    input = torch.randint(0, 255, size=(4, 1, 28, 28))
    model = CNNModel()
    assert model.forward(input.float()).shape == (
        input.shape[0],
        10,
    ), f"Output format of model is not ({input.shape[0]},10)"


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expect 4D tensor"):
        model = CNNModel()
        model.forward(torch.randn(1, 2, 3))


def test_channels():
    with pytest.raises(
        ValueError, match="Expected channel size not equal to 1"
    ):
        model = CNNModel()
        model.forward(torch.randn(4, 2, 3, 3))
