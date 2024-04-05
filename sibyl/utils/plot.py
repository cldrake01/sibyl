import os

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from sibyl.utils.config import Config
from sibyl.utils.log import find_root_dir


def pred_plot(
    X: Tensor,
    y: Tensor,
    y_hat: Tensor,
    loss: list,
    config: Config,
    features: list[int] | None = None,
):
    """
    Plot both the predicted vs. actual values and the loss on the same graph.

    :param X: The context window which precedes the target window.
    :param y: The target window.
    :param y_hat: The predicted target window.
    :param loss: List of loss values.
    :param config: TrainingConfig object.
    :param features: List of features to plot.
    """
    if not config.plot_predictions and not config.plot_loss:
        return

    criterion = config.criterion.__class__.__name__

    # Clear existing figure
    plt.clf()

    sns.set_theme(style="dark")

    # First subplot for predicted vs actual
    if config.plot_predictions:
        # 1 row, 2 columns, 1st subplot
        sub = plt.subplot(1, 1, 1)
        # sub.xaxis.set_label_position("top")
        sub.xaxis.tick_top()
        sub.yaxis.tick_left()
        X_, y_, y_hat_ = (
            X.detach().squeeze(0),
            y.detach().squeeze(0),
            y_hat.detach().squeeze(0),
        )

        features = features or range(X_.shape[1])

        for i in features:
            actual = pd.DataFrame(torch.cat((X_[:, i], y_[:, i]), 0))
            predicted = pd.DataFrame(torch.cat((X_[:, i], y_hat_[:, i]), 0))
            sns.lineplot(
                data=actual,
                palette=["blue"],
                alpha=0.5,
            ).legend().remove()
            sns.lineplot(
                data=predicted,
                palette=["red"],
                alpha=0.5,
            ).legend().remove()

    # Second subplot for loss
    if config.plot_loss:
        # 1 row, 2 columns, 2nd subplot
        if config.plot_predictions:
            sub = plt.subplot(2, 1, 2)
            # Place a label on the right side of the plot
            # sub.yaxis.set_label_position("right")
            sub.xaxis.tick_bottom()
            sub.yaxis.tick_right()
        else:
            plt.subplot(1, 1, 1)
        # Make its background transparent
        plt.gca().patch.set_alpha(0.0)
        # Make it borderless
        plt.gca().spines["top"].set_alpha(0.0)
        # Plot the moving average of the loss using a pandas dataframe
        sns.lineplot(
            data=pd.DataFrame(loss).rolling(config.plot_interval).mean(),
            palette=["g"],
            alpha=0.5,
        ).legend().remove()

    path: str = find_root_dir(os.path.dirname(__file__))
    path += "/assets/plots/forecasts/"
    path += f"{config.dataset_name}/"
    os.makedirs(path, exist_ok=True)
    path += f"{criterion}.png"

    plt.savefig(
        path,
        dpi=500,
    )
    plt.show()


def bias_variance_plot(
    *lists: list[float],
    step: str,
    config: Config,
):
    criterion = config.criterion.__class__.__name__

    sns.set_theme(style="dark")

    lists = [pd.DataFrame(lst).rolling(config.plot_interval).mean() for lst in lists]

    for lst in lists:
        plt.plot(lst, alpha=0.5)
        config.log.metric(
            f"{config.dataset_name} - {criterion} - {step} | {lst.mean().item():.2f}"
        )

    plt.title(
        f"{criterion} {step} Bias-Variance Decomposition at Epoch {config.epoch}\n"
        + " | ".join([f"{lst.mean().item():.5f}" for lst in lists])
    )

    path: str = find_root_dir(os.path.dirname(__file__))
    path += "/assets/plots/bias-variance/"
    path += f"{config.dataset_name}/"
    os.makedirs(path, exist_ok=True)
    path += f"{step}-{criterion}-{config.epoch}.png"

    plt.savefig(
        path,
        dpi=500,
    )
    plt.show()
