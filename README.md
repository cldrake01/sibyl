# Sibyl

**Authors:** Collin Drake, Jack Cerullo  
**Affiliation:** Peak to Peak Charter School  
**Date:** March 2024

## Abstract

This paper addresses the issue of biased and underfitting predictions in transformer-based time series forecasting models. It identifies the mean component in traditional loss functions like Mean Squared Error (MSE) and Mean Absolute Error (MAE) as the source of bias, leading to predictions lacking in volatility. To mitigate this, the paper proposes two variance-weighted loss functions, Variance-weighted Maximum Squared Error (VMaxSE) and Variance-weighted Maximum Absolute Error (VMaxAE), which prioritize capturing the shape of the time series over averaging. Experimental results demonstrate the effectiveness of these loss functions in improving prediction accuracy and volatility while reducing bias.

## Introduction

The paper discusses the recent surge in efforts to adapt transformers for time series forecasting tasks. Despite advancements, transformer models exhibit a tendency towards underfitting, resulting in biased and linear predictions regardless of architecture. This underfitting is attributed to the mean component in traditional loss functions like MSE and MAE.

## Methodology

The paper introduces two alternative loss functions, VMaxSE and VMaxAE, which incorporate variance-weighting to prioritize capturing the shape of the time series. These loss functions aim to mitigate bias by penalizing deviations from the actual sequence's variance.

## Results

Experimental results demonstrate that the proposed loss functions, VMaxSE and VMaxAE, lead to predictions with improved volatility and accuracy compared to traditional loss functions. Visualizations illustrate the effectiveness of the proposed approach in capturing the shape of the time series.

## Conclusion

In conclusion, the paper highlights the necessity of rethinking loss functions in time series forecasting, particularly in transformer-based models. The proposed variance-weighted loss functions offer a promising solution to address bias and underfitting, paving the way for more accurate predictions.

## Acknowledgements

The authors acknowledge the support of their teachers, Mr. Robert Hettmansperger, Mr. Jake Lehr, and Mx. Seonjoon-young, in conducting this research.

## References

The paper provides references to datasets used for experimentation, including links to relevant repositories and data sources.

