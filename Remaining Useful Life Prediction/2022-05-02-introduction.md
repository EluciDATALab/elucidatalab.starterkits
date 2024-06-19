---
title: Introduction
Featured_Image: ../Remaining Useful Life Prediction/img/SK_specific/RUL_figure1.png
---

## Introduction
<br/>
<p align="center"><iframe src="https://player.vimeo.com/video/595335197?h=51878aaf4c&color=e700ef" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>
<br/></p>
<br/>
Welcome to the tutorial for the AI Starter Kit on remaining useful lifetime prediction!
In this first video we will provide an overview of the actual business case we’re tackling. We will introduce the most important concepts and explain why it is beneficial for maintenance to know the remaining useful lifetime of a machine, a tool or any other industrial asset.

Estimating the remaining useful lifetime of an asset can be beneficial for several purposes in a variety of industrial contexts. For one, it can offer support in the better scheduling of maintenance operations, for example for offshore wind turbines. For such assets, maintenance can only be performed under the right weather conditions.

But also for other assets, maintenance is an important part of its lifecycle. Traditionally, maintenance is done in a corrective way, such that it is performed at the moment an asset has failed. In that case, the failing part is identified and rectified or replaced, to ensure that the asset can subsequently resume normal operation.

Especially for strongly interdependent production lines or industrial assets operating in critical environments, an unplanned downtime is to be avoided and often costly. For this reason, more recently, for an increasing number of assets preventive maintenance is performed. In that case, maintenance tasks are scheduled at regular intervals, avoiding future asset failure to a maximum extent – but with the risk to replace assets that are still working correctly and still could for a while. Hence, the optimal time for replacement is wanted: The time when the asset still works correctly but will probably fail soon. And this is where the remaining useful lifetime comes into play.

Nowadays, ever more assets are equipped with different types of sensors gathering data.  This started the trend towards predictive maintenance, in which the maintenance moments are decided upon in a data-driven way. Therewith, maintenance is only performed when actually needed. This is a very broad domain which encompasses a variety of topics. Besides the estimation of the asset’s remaining useful lifetime – the main topic of this AI starter kit – further related problems to solve for a fully predictive maintenance approach are failure prediction, detection, and diagnosis for root cause analysis.

The remaining useful lifetime of industrial assets is defined as an estimate of the remaining time that an item, component, or system is estimated to function as expected. It is expressed as the number of hours, cycles, batches or any other quantity.

<center><img src="./img/SK_specific/RUL_figure1.png" width="500" class="center" /></center>

In the figure we see datapoints collected from an arbitrary sensor in green. Let’s say it measures the tool radius of a milling machine. With increasing time, the radius decreases as the tool wears out. At a certain point, the tool is considered too worn out to still further use it in production, which can influence the quality of the produced parts or lead to an unstable production process and consequent damage to the machine. The main questions thus is for how long the tool will it still be able to function properly?

By applying Machine Learning algorithms, the continuation of this time series data can be forecasted, denoted in pink. From this forecast, the remaining useful lifetime can be estimated for a given asset. In this case, the machine should be maintained within the next 10 to 15 cycles. With this information, maintenance tasks can be scheduled accordingly – with a tool still functioning and without unexpected downtime. Hence, by estimating the remaining useful lifetime, we minimize the risk of failure and maintenance cost.

An accurate prediction of an asset’s lifetime can also help to optimise the operational efficiency of an asset, by more optimally planning the use of that asset before its end of life, or adapting its operational use to extend its useful lifetime. In manufacturing settings, where a downtime of the production line results in several operational problems and associated costs, remaining useful lifetime prediction can avoid unplanned downtime.

In this AI Starter Kit, we will show its use to avoid safety-critical situations, by illustrating the solution methodology for predicting the remaining useful lifetime of aircrafts engines.

To summarize, by applying predictive maintenance it is possible to:
- better schedule maintenance operations
- optimize operational efficiency
- avoid unplanned downtime
- anticipate and avoid safety-critical situations.

This analysis can be performed for various machines, tools or processes as long as this degradation over time can be measured accordingly, just like resistance, length, temperature or similar measures.

Predicting the remaining useful lifetime is typically very challenging for several reasons:
First of all, usually a multitude of heterogenous sensor data is measured at several places in the machine. This data is often captured at a high-frequency, consisting of vibration data, acoustic emission, accelerometer data, and many more machine-internal parameters. Secondly, in some settings, also characteristics of the surrounding environment influencing the lifetime of the asset are captured. Further, typically, data that is gathered over a long period of time needs to be available to train the model. Given that industrial assets are often very complex – as they consist of both electrical and mechanical components - it generally is a non-trivial exercise to predict their end of life. On top of that, the data typically comes with a lot of variation, due to varying machine types operating in heterogenous operating conditions with different machine configurations.

<center><img src="./img/SK_specific/RUL_figure2.png" width="800" class="center" /></center>

Therefore, the main challenge in this process is to extract meaningful characteristics that can be used to predict the end of life from the gathered data. This is complicated by the fact that usually very little domain expertise on the operating conditions of these assets is available. Additionally, the environment in which they operate is typically very dynamic.

In this AI Starter Kit, we will guide you through a data-driven methodology based on deep learning to tackle this challenge.
In the next video, we will concentrate on the data itself that we selected to illustrate the approach. We will explain which information is available and what we can learn from it.
