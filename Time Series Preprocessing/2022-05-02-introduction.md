---
title: Introduction
Featured_Image: ../Time Series Preprocessing/img/SK_specific/TSP_figure7_introduction.png
---

## Introduction
<br/>
<p align="center"><iframe src="https://player.vimeo.com/video/596619071?h=ccbb7e33ba&color=e700ef" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>
<br/></p>
<br/>

Welcome to the tutorial for the AI Starter Kit on time-series pre-processing. In this first video we will provide an overview of the actual challenge we’re tackling. We will introduce the most important problems occurring in time series data and explain why it is beneficial to first improve your data before you start any other data-modelling tasks.

Time series are characterised as a collection of data points obtained at successive times, most often with equal intervals between them. Since an increasing amount of assets is instrumented with sensors, this type of data is omnipresent, for example when monitoring industrial machinery or resulting from wearables tracking one’s health state. Depending on the application domain, time series data is exploited for various purposes. It is used, amongst others, for profiling energy consumption of households predicting imminent failures of a production line, estimating the remaining useful lifetime of a machine, and many more industrial use cases.

Typically, time series data cannot be used easily as-is by machine learning algorithms. This can be for example due to data quality issues such as missing values and sensor misreadings. Or because some algorithms are not fit to deal with the continuous nature of this type of data or the associated - often high – frequency with which it is gathered.

To illustrate the various methods and techniques in this Starter Kit, we will use a publicly available real-world dataset that consists of sensor data measurements from wind turbines. It exhibits typical characteristics of industrial time series datasets as it is very noisy and contains outliers and missing values.

Moreover, it exhibits seasonal patterns across multiple years, as the machine behaviour is influenced by the meteorological conditions.

We will use this dataset to illustrate
* how resampling and smoothing can be applied to gain a better understanding of the behaviour of the time series,
* how the quality of the data can be improved through normalization and outlier detection, and
* how missing data can be imputed in various ways.

In the next video, we will perform some initial exploration on the dataset in order to gain some basic understanding of the data.
