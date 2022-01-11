import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.models.predict_model import loader
from visualization import _PROJECT_ROOT


def visualize():
    model, predictions, _, _ = loader()
    data = TSNE(
        n_components=2, learning_rate="auto", init="random"
    ).fit_transform(predictions.detach().numpy())
    plt.plot(data[:, 0], data[:, 1], "x")
    plt.savefig(f"{_PROJECT_ROOT}/reports/figures/vis_last_lin_layer.png")


if __name__ == "__main__":
    visualize()
