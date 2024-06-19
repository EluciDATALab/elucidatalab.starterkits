---
title: Introduction
Featured_Image: ../Federated Learning/img/SK_specific/FedRepo.png
---

## Introduction
<!-- <br/>
<p align="center"><iframe src="https://player.vimeo.com/video/612907452?h=1c07951c12&color=e700ef" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>
<br/></p>
<br/> -->
### Description

Forecasting electricity consumption accurately is crucial for managing energy resources efficiently, ensuring reliable power supply, and optimizing grid operations. This task is complex, particularly because of the diverse and evolving usage patterns across different households. Usage patterns vary not only across households, but also over time (e.g. number of occupants changes). This leads to what is known as concept drift, which occurs when the underlying data patterns that predictive models have learned change, potentially degrading the model’s performance.

In scenarios where data privacy is paramount and bandwidth is limited, the concept of federated learning emerges as a vital approach. Federated learning allows multiple decentralized entities—such as different households in the case of electricity consumption—to collaboratively learn a predictive model while keeping all training data local. This method avoids the need to transfer large volumes of sensitive data to a central server, addressing significant privacy concerns. However, federated learning environments face significant challenges among which concept drift is a prominent one:

 - Dynamic and heterogeneous data sources often lead to concept drift, where the underlying data patterns the model has learned change over time, potentially degrading the model's performance.
 - Limited bandwidth for communication between clients and the central server can hinder the efficiency of model updates and retraining processes.
 - Privacy concerns limit the amount and type of data that can be shared between clients, complicating the detection and mitigation of concept drift.

These challenges necessitate advanced strategies for model training and maintenance to ensure that predictive models remain accurate and efficient over time without compromising privacy or incurring prohibitive communication costs.


### Business goal

The business goal for this Starter Kit is **electricity forecasting** in federated learning environments. Specifically, a **concept drift mitigation** strategy will be applied to cope with the occurence of concept drift due to the heterogeneous and dynamic nature of electricity usage data. The methodology, called *FedRepo*, was introduced by Tsiporkova et al. [1] and aims to provide a robust solution that maintains the accuracy and efficiency of the federated models over time, ensuring they adapt to changes in data dynamics while minimizing communication overhead.

### Application context

The FedRepo methodology is applicable in various settings where data privacy and limited connectivity are major concerns:

 - Healthcare: Hospitals and medical institutions can collaborate on developing predictive models or improving diagnostic tools on more diverse data without actually sharing patient data.
 - Wearables: User experience of several features (e.g. text prediction) can be enhanced on personal devices without compromising privacy.
 - Industrial: Assets manufactured by a third party (e.g. printers) can be used to collaborately learn predictive models without each customer having to share its data. For example, data from various printers in different settings can be used to predict maintenance needs or optimize performance without exposing individual usage patterns or sensitive business information.

### Starter Kit outline

We will first describe the dataset that we will be using for this Starter Kit, which consists of accelerometer data containing six time series signals. Then, we conceptually explain the Swinging Door Trending technique, and apply this on the six signals separately. Afterwards, we apply the technique on the six signals at once, which makes sense since accelerometer signals are often correlated. We then evaluate the results of both compression approaches on a technical level by considering how much they manage to compress the original signals and what is the associated error. Finally, we validate the compression within a concrete application, i.e. event detection.

### Dataset

The forecasting of electricity consumption across households is a highly relevant application for this methodology as energy consumption of households obviously is privacy-sensitive. This was demonstrated in [2], where it is highlighted how household energy patterns can reflect socio-economic statuses. Additionally, many factors could cause for concept drift to occur:

 - The occupation of the household in terms of its inhabitants
 - Replacement of household appliances
 - Fluctuations in electricity prices, encouraging more conservative usage during high price periods
 - ...

The data used is collected by the UK Power Networks led Low Carbon London project ([available here](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)). It consists of 5,567 households (given in column `consumer`) in London representing a balanced sample representative of the Greater London population with a 30-minutes granularity between November 2011 and February 2014. The consumption (in column `consumption`) is given in kWh. For demonstrating our methodology, we randomly selected 300 households for which we ensured that the data is available until at least 01/2014. For these households, a repository of federated models will be trained in order to forecast the consumption within the next 30 minutes.

[1] Tsiporkova, E., De Vis, M., Klein, S., Hristoskova, A., & Boeva, V. (2023). Mitigating Concept Drift in Distributed Contexts with Dynamic Repository of Federated Models. In 2023 IEEE International Conference on Big Data (BigData) (pp. 2690-2699). IEEE.

[2] Christian Beckel, Leyna Sadamori, and Silvia Santini. 2013. Automatic socio-economic classification of households using electricity consumption data. In Proceedings of the fourth international conference on Future energy systems (e-Energy '13). Association for Computing Machinery, New York, NY, USA, 75–86. https://doi.org/10.1145/2487166.2487175

[3] Omran, M.G.H., Salman, A. & Engelbrecht, A.P. Dynamic clustering using particle swarm optimization with application in image segmentation. Pattern Anal Applic 8, 332–344 (2006). https://doi.org/10.1007/s10044-005-0015-5
