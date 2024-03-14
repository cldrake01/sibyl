import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor
import seaborn as sns

from sibyl.utils.configs import TrainingConfig
from sibyl.utils.log import find_root_dir


def plot(
    X: Tensor,
    y: Tensor,
    y_hat: Tensor,
    loss: list,
    config: TrainingConfig,
    features: list[int] = None,
):
    """
    Plot both the predicted vs actual values and the loss on the same graph.

    :param X: The context window preceding the target window.
    :param y: The target window.
    :param y_hat: The predicted target window.
    :param loss: List of loss values.
    :param config: TrainingConfig object.
    :param features: List of features to plot.
    """
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
            sns.lineplot(
                data=pd.DataFrame({"y": torch.cat((X_[:, i], y_[:, i]), 0)}),
                palette=["b"],
                alpha=0.5,
            ).legend().remove()
            sns.lineplot(
                data=pd.DataFrame({"y_hat": torch.cat((X_[:, i], y_hat_[:, i]), 0)}),
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

    # Show the combined plot
    if config.plot_predictions or config.plot_loss:
        # Save the latest plot
        plt.savefig(
            f"{find_root_dir(os.path.dirname(__file__))}/assets/plots/latest.png",
            dpi=300,
        )
        plt.show()
