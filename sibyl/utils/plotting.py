import os

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from sibyl.utils.configuration import Config
from sibyl.utils.logging import find_root_dir


def predicted_vs_actual(
    x: Tensor,
    y: Tensor,
    y_hat: Tensor,
    loss: list,
    config: Config,
    features: list[int] | None = None,
    filename: str | None = None,
):
    """
    Plot both the predicted vs. actual values and the loss on the same graph.

    :param x: The context window which precedes the target window.
    :param y: The target window.
    :param y_hat: The predicted target window.
    :param loss: List of loss values.
    :param config: TrainingConfig object.
    :param features: List of features to plot.
    :param filename: The filename to save the plot as.
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
            x.detach().squeeze(0),
            y.detach().squeeze(0),
            y_hat.detach().squeeze(0),
        )

        features = features or range(X_.shape[1])

        for i in features:
            actual = pd.DataFrame(torch.cat((X_[:, i], y_[:, i]), 0))
            predicted = pd.DataFrame(torch.cat((X_[:, i], y_hat_[:, i]), 0))
            sns.lineplot(
                data=predicted,
                palette=["red"],
                alpha=0.50,
            ).legend().remove()
            sns.lineplot(
                data=actual,
                palette=["blue"],
                alpha=0.50,
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
    path += filename or f"{criterion}.png"

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

    if not config.plot_interval:
        config.log.warning("Plotting is disabled at `plot_interval = 0`.")
        return

    for metric, values in config.metrics.items():
        plt.plot(
            pd.DataFrame(values).rolling(config.plot_interval).mean(), label=metric
        )

        plt.title(
            f"{config.dataset_name} - {config.criterion.__class__.__name__}"
            f"- {config.stage} Metrics at Epoch {config.epoch}"
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


def metrics_table(metrics_: list[tuple[str, pd.DataFrame]], dataset: str):
    """
    Plot a table of metrics.

    :param metrics_: The metrics to plot.
    :param dataset: The dataset name.
    """
    # Each data frame is structured as such:
    #           bias  variance       mse       mae
    # 0     0.841117  0.800877  1.490557  1.038116
    # 1     0.891275  0.937894  1.711424  1.124250
    # 2     0.808641  0.786355  1.422781  1.015115
    # 3     0.955522  0.841266  1.735593  1.131148
    # 4     0.831139  0.775454  1.449014  1.014566
    # ...        ...       ...       ...       ...
    # 3874  0.924198  0.822155  1.658028  1.099741
    # 3875  0.911048  0.832099  1.643615  1.102134
    # 3876  0.896106  0.780433  1.566096  1.067901
    # 3877  0.845971  0.789652  1.487770  1.043889
    # 3878  0.825264  0.765414  1.429466  1.012620
    #
    # [3879 rows x 4 columns]
    names = [name for name, _ in metrics_]

    aggregated = [
        [
            sum(x := metrics_[i][1][col][-1_000:]) / len(x)
            for col in metrics_[0][1].columns
        ]
        for i in range(len(metrics_))
    ]

    # Round the values to 5 decimal places
    aggregated = [[round(val, 5) for val in row] for row in aggregated]

    # Aggregated is structured such that it is a list of lists where 0 corresponds to the bias, 1 to the variance, etc.
    # Instead, we want to transpose this list so that each list corresponds to a row in the table.
    table = [names] + list(map(list, zip(*aggregated)))

    # Create a table
    fig: go.Figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", *metrics_[0][1].columns],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=table,
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    path: str = find_root_dir(os.path.dirname(__file__))
    path += f"/assets/plots/tables/"
    path += f"{dataset}/"
    os.makedirs(path, exist_ok=True)
    path += "-".join(names)
    path += ".png"

    fig.write_image(path)
