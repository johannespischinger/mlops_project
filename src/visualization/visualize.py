import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from src.models.predict_model import loader
from src.models.train_model import CNNModel


def visualize():
    model, predictions, _, _ = loader()
    data = TSNE(
        n_components=2, learning_rate="auto", init="random"
    ).fit_transform(predictions.detach().numpy())
    plt.plot(data[:, 0], data[:, 1], "x")
    plt.savefig(
        "/Users/johannespischinger/Documents/Uni/Master/Erasmus/Courses/MLOps/cookiecutter-ds-project/reports/figures/vis_last_lin_layer.png"
    )


if __name__ == "__main__":
    visualize()
