---
title: Key Take Away Messages
Featured_Image: ../Time Series Preprocessing/img/general_take_home.png
collab: https://colab.research.google.com/github/EluciDATALab/elucidatalab.starterkits/blob/main/notebooks/SK_3_4_Time_Series_Preprocessing/elucidata_starterkit_3_4.ipynb
---

## Key Take Away Messages
<br/>
<p align="center"><iframe src="https://player.vimeo.com/video/596618633?h=a324567dda&color=e700ef" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>
<br/></p>
<br/>

In the video tutorial for this AI Starter Kit, we have demonstrated techniques that can be used to preprocess time series data in order to improve its quality for further exploitation. To illustrate the respective methods and techniques, we have used a real-world dataset containing wind turbine data that exhibits the typical characteristics of industrial datasets, including noise, outliers, missing values, and seasonal patterns.

First, we demonstrated how resampling and smoothing can be applied to better understand the behaviour of the time series. As we have shown, resampling techniques allow to reduce the temporal granularity, thereby revealing longer-term trends and hiding sharp, fast fluctuations in the signal. Smoothing on the other hand modifies the time series data such that individual points that are higher than the adjacent points - presumably caused by noise - are reduced, and points that are lower than the adjacent points are increased leading to a smoother signal.

Subsequently, we showed how the quality of the data can be improved through normalization and outlier detection. Normalization rescales the range of a particular variable in order to make different variables with different ranges comparable. Outlier detection tried to identify the extreme values in the time series that deviate from other observations in the data. The detected outliers can subsequently be examined in more detail to investigate whether these were caused by measurement errors, in which case they can be removed, or if it concerns unexpected phenomena which are interesting to look into in more detail.

Finally, another frequently occurring problem with time series data is that it suffers from missing data, caused by for example senor malfunctioning, communication errors, or due to outliers that were removed. We illustrated how to impute this missing data by means of 3 different techniques: linear interpolation, fleet-based imputation and pattern-based interpolation, and outlined what are the advantages and disadvantages of these techniques.

The use of these methods allows to improve the quality of the data and to prepare it for further exploration, analysis and modelling purposes. This is a crucial step, as it will improve the completeness and reliability of the input data, and consequently the accuracy of your results.

We thank you for completing this video series and hope to welcome you in another AI Starter Kit tutorial.
