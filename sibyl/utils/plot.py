import os

import pandas as pd
import seaborn as sns
import torch
from great_tables import GT
from matplotlib import pyplot as plt
from torch import Tensor

from sibyl.utils.config import Config
from sibyl.utils.log import find_root_dir


def predicted_vs_actual(
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


def bias_variance(
    *lists: list[float],
    config: Config,
) -> tuple[float, ...]:
    """
    Plot the bias-variance decomposition of the loss function.

    :param lists: The lists of bias and variance values.
    :param config: The configuration object.
    """
    criterion = config.criterion.__class__.__name__

    sns.set_theme(style="dark")

    lists = tuple(
        pd.DataFrame(lst).rolling(config.plot_interval).mean() for lst in lists
    )
    means = tuple(lst[: config.plot_interval].mean().item() for lst in lists)

    for l, m in zip(lists, means):
        plt.plot(l, alpha=0.5)
        config.log.metric(
            f"{config.dataset_name} - {criterion} - {config.stage} | {m:.2f}"
        )

    plt.title(
        f"{criterion} {config.stage} Bias-Variance Decomposition at Epoch {config.epoch}\n"
        + " | ".join([f"{lst.mean().item():.5f}" for lst in lists])
    )

    path: str = find_root_dir(os.path.dirname(__file__))
    path += "/assets/plots/bias-variance/"
    path += f"{config.dataset_name}/"
    os.makedirs(path, exist_ok=True)
    path += f"{config.stage}-{criterion}-{config.epoch}.png"

    plt.savefig(
        path,
        dpi=500,
    )
    plt.show()

    return means


def metrics(config: Config):
    """
    Plot the metrics.

    :param config: The configuration object.
    """
    sns.set_theme(style="dark")

    for metric, values in config.metrics.items():
        plt.plot(
            pd.DataFrame(values).rolling(config.plot_interval).mean(), label=metric
        )
        path: str = find_root_dir(os.path.dirname(__file__))
        path += f"/assets/plots/{metric}/"
        path += f"{config.dataset_name}/"
        os.makedirs(path, exist_ok=True)
        criterion = config.criterion.__class__.__name__
        path += f"{config.stage}-{criterion}-{config.epoch}.png"

        plt.savefig(
            path,
            dpi=500,
        )

        plt.show()


def metrics_table(config: Config):
    """
    Plot a table of metrics.

    :param config: The configuration object.
    """
    sns.set_theme(style="dark")

    # Create a dataframe of `loss-function`, `bias`, and `variance` metrics

    # Average each metric
    metrics = (
        f"{pd.DataFrame(metric).mean().item():.5f}"
        for metric in config.metrics.values()
    )
    # Create a dataframe
    df = pd.DataFrame(
        {
            "Metric": list(config.metrics.keys()),
            "Value": metrics,
        }
    )

    # Plot the table
    name = config.criterion.__class__.__name__
    path = find_root_dir(os.path.dirname(__file__))
    path += f"/assets/plots/tables/{config.dataset_name}/"
    os.makedirs(path, exist_ok=True)
    path += f"{name}.png"

    GT(df).tab_header(
        title=name,
        subtitle=" | ".join(f"{m}" for m in tuple(config.metrics.keys())),
    ).opt_style(style=1, color="blue").save(path)
