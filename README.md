# Sibyl

## Overview

There's an enormous amount of boilerplate associated with financial
time series forecasting. Sibyl aims to simplify the process by providing a
simple, modular, and extensible framework for financial time series forecasting.
In brief, Sibyl provides interfaces for data loading, model training, and inference.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quick Start Guide](#quick-start-guide)
- [Usage](#usage)
    - [Data Gathering & Preprocessing](#data-gathering--preprocessing)
    - [Normalization](#normalization)
    - [Windowing](#windowing)
    - [Configuration](#configuration)
- [Sibyl's Components](#sibyls-components)
- [Contributing](#contributing)
- [License](#license)

## Usage

### Data Gathering & Preprocessing

APIs tend to return time series data in a variety of formats.
For equities, some APIs return data like these:

```json
{
  "AAPL": {
    "2021-01-01": 100.0,
    "2021-01-02": 101.0,
    "2021-01-03": 102.0
  }
}
```

Others may return data like these:

```json
{
  "AAPL": [
    {
      "date": "2021-01-01",
      "price": 100.0
    },
    {
      "date": "2021-01-02",
      "price": 101.0
    },
    {
      "date": "2021-01-03",
      "price": 102.0
    }
  ]
}
```

Sibyl's interface accepts a function to load time series data.

### Normalization

In our approach, we normalize time series data using a specialized formula implemented in PyTorch:

```py
torch.sign(tensor) * torch.log10(torch.abs(tensor) + 1.0)
```

Expressed in LaTeX, the normalization formula is:

$$
\text{sign}(x) \cdot \log_{10}(|x| + 1)
$$

Rationale Behind This Normalization Technique:

- **Preventing Future Data Leakage:** In time series forecasting, it's imperative to avoid any indirect leakage of
  future
  data. Traditional normalization methods like Z-score and min-max normalization can inadvertently reveal information
  about the entire series through their rescaling process, thus compromising the model's ability to generalize from past
  data without prior knowledge of future data.

- **Handling Negative Values:** Conventional logarithmic normalization methods suggest adding a constant to adjust for
  negative values, leading to an asymmetrical distribution where negative values dominate the lower logarithmic range,
  and
  positive values occupy the higher range. This approach skews the data distribution, making it challenging to maintain
  a
  balanced representation of the original series' dynamics.

Our chosen method addresses these challenges effectively:

- **Sign Preservation:** By applying the sign function, we ensure that the normalized values retain their original sign,
  maintaining the distinction between positive and negative fluctuations in the data.

- **Symmetrical Distribution:** The logarithmic transformation of the absolute values, offset by one, ensures a
  symmetrical
  distribution of both positive and negative values around the origin. This balanced approach facilitates a more
  accurate
  representation of the data's inherent characteristics, without the biases introduced by traditional normalization
  techniques.

### Windowing

Sibyl creates a windowed dataset from the normalized time series data.
The windowed dataset is a supervised learning problem where the target `y`
is the next `target_window_size` values in the time series and the input `X` is a window of
length `feature_window_size` of the time series data.

### Configuration

Sibyl uses several configuration classes to manage the forecasting process:

1. **TimeSeriesConfig**: This class manages the configuration of the time series data.
2. **TrainingConfig**: This class manages the configuration of the training process.

See [__init__py](sibyl/__init__.py).

## Sibyl's Components

Sibyl is composed of several components:

1. **Data**: Sibyl provides a simple interface for loading time series data.
2. **Models**: Sibyl provides a simple interface for training and evaluating forecasting models.
3. **Metrics**: Sibyl provides a simple interface for evaluating forecasting models.
4. **Utilities**: Sibyl provides a simple interface for common time series operations.
5. **Logging**: Sibyl provides a simple interface for logging.

**Note:** [environment.yml](environment.yml) is used by the Docker image build process
and can be created using `conda env export > environment.yml`.

**Note:** the [logs](logs) directory referenced by the `logger` function is excluded from Git.