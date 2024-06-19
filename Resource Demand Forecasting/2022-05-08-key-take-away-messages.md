---
title: Key Take Away Messages
Featured_Image: ../Resource Demand Forecasting/img/general_take_home.png
collab: https://colab.research.google.com/github/EluciDATALab/elucidatalab.starterkits/blob/main/notebooks/SK_1_3_Resource_Demand_Forecasting/elucidata_demonstrator_1_3.ipynb
---

## Key Take Away Messages
In the video tutorial for this AI Starter Kit, we demonstrated forecasting the demand for a particular resource at a particular point in the future, illustrated on the case of energy demand forecasting. In doing so, we have taken you through the different steps in the data science workflow.

{% vimeo 600085743?h=492077c656 [500] [200] %}

We first performed a number of required data preprocessing steps in order to prepare the data for further analysis. This included fusing the electricity and weather data, requiring to resampling the former data to a lower sampling rate.

Subsequently, we explored the fused dataset in order to better understand which factors influence the household energy consumption. To this end, we resorted to both visual and statistical exploration methods. Based on this, we could identify a number of yearly, weekly, and daily patterns, including particular behavior during weekends and some holiday periods. This information formed the basis for the feature extraction that followed next, such as the energy consumption a week before, the day of the week and the hour of the day.

Finally, we served the extracted features to two different models, namely Random Forest Regression and Support Vector Regression to perform the actual forecasting. Through different experiments, we analyzed the influence of the training strategy, type of machine learning model and effect of data normalization on the model performance. Based on this, it became clear that achieving good prediction results depends on different factors, including the appropriate data preprocessing, the amount of available training data and the algorithm parametrisation.

As next steps, we invite you to further explore each of these influences in more detail. Especially with respect to the hyperparameter tuning, it proves worth to further explore the different parameter settings and study how these changes impact the model performance. Furthermore, you can also try to adapt the different steps to the context of your own dataset.

While the details of each of the steps might differ, the methodological steps we presented are typically the required phases you need to go through when solving a resource demand forecasting problem.

We thank you for completing this video series and hope to welcome you in another AI Starter Kit tutorial.

## Additional information

The video material in this website was developed in the context of the [SKAIDive project](https://elucidata.be/skaidive), financially supported by the [European Social Fund](https://www.esf-vlaanderen.be), the European Union and Flanders. For more information, please contact us at <elucidatalab@sirris.be>
