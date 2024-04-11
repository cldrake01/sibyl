# Sibyl

---

This repository provides the code for *Advancing Time Series Forecasting: Variance-Aware Loss Functions in Transformers*. 
The paper can be found either within this [folder](/paper) or at ...   

---

## Abstract

When forecasting time series data, transformer models predict sequences lacking in volatility, exhibiting significant bias. We hypothesize that transformer models do so because of their loss functions. More specifically, we posit that the mean component of mean squared error and mean absolute error cause this behavior. We propose two alternative loss functions – Variance-weighted Maximum Squared Error and Variance Weighted Absolute Error – which, crucially, do not incorporate averaging and output variance in the error calculation. We do so to prevent our transformer from converging at a minimum wherein it reduces loss by merely forecasting a time series devoid of volatility, helping time series transformer models continue to train without the risk of underfitting towards the mean. PyTorch implementations of the models used in this project can be found at [github.com/cldrake01/sibyl](https://github.com/cldrake01/sibyl).

---