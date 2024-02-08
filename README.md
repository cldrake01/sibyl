# Sibyl

## Overview

There's an enormous amount of boilerplate associated with time series forecasting.
Sibyl aims to simplify the process by providing a simple, modular,
and extensible framework for time series forecasting.

## Sibyl's Assumptions

Sibyl assumes several things about your data and your intentions:

1. The data are time series data.
2. You intend to forecast future values of the time series.
3. The forecasting horizon is fixed and known in advance.

Sibyl does not assume:

1. The data are univariate.
2. The data are regularly spaced.
3. The data are stationary.
4. The data are normally distributed.
5. The data are free of missing values.
6. The data are free of outliers.

## Methodology

### Normalization

Sibyl normalizes the time series data using the following formula:

```py
torch.sign(tensor) * torch.log10(torch.abs(tensor) + 1.0)
```

Which can be expressed in LaTeX as:

$$
\text{sign}(x) \cdot \log_{10}(|x| + 1)
$$

Why this form of normalization?

Firstly, in any time series forecasting task, 
it's crucial to avoid leaking future 
data indirectly. Z-score normalization
and min-max are both highly error prone,
as they rescaled data themselves
reveal information about the entire time
series.

Secondly, if your data comprise negative
values ordinary logarithmic normalization
techniques would suggest shifting 



### Windowing

Sibyl creates a windowed dataset from the normalized time series data.
The windowed dataset is a supervised learning problem where the target `y`
is the next `target_window_size` values in the time series and the input `X` is a window of
length `feature_window_size` of the time series data.

## Sibyl's Components

Sibyl is composed of several components:

1. **Data**: Sibyl provides a simple interface for loading time series data.
2. **Models**: Sibyl provides a simple interface for training and evaluating forecasting models.
3. **Metrics**: Sibyl provides a simple interface for evaluating forecasting models.
4. **Utilities**: Sibyl provides a simple interface for common time series operations.
5. **Logging**: Sibyl provides a simple interface for logging.

**Note:** [environment.yml](environment.yml) is used by the Docker image build process
and can be created using `conda env export > environment.yml`.

**Note:** the [logs](logs) and [notebooks](notebooks) directories remain local and are not pushed to GitHub.
