---
title: Key Take Away Messages
Featured_Image: ../Remaining Useful Life Prediction/img/general_take_home.png
collab: https://colab.research.google.com/github/EluciDATALab/elucidatalab.starterkits/blob/main/notebooks/SK_1_2_1_Remaining_Useful_Life_Prediction/elucidata_starterkit_1_2_1.ipynb
---

## Key Take Away Messages
<br/>
<div align="center"><iframe width="600" height="336" src="https://www.youtube.com/embed/6yr0thvqMI0" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>
<br/>

In the video tutorial for this AI Starter Kit, we covered the basics of using deep learning for remaining useful lifetime prediction, which is one of the central topics in predictive maintenance. We illustrated the approach on the use case of predicting the remaining useful life of aircraft engines based on run-to-failure data. In doing so, we have taken you through the different steps in the data science workflow.

We started with the business and data understanding, which allowed us to gain more insights into the specific problem to solve and the structure of the data. Based on this information, we continued with the data preprocessing, in which we explained how to appropriately prepare the data. This included the construction of a training, test and validation set to learn and evaluate a model to predict whether an engine will fail within a certain number of operational cycles.

In the video on data modelling and analysis, we showed you how to use deep learning to construct a model for remaining useful life prediction. In particular, we applied a deep learning approach called long short-term memory (LSTM) networks which allows the construction of a predictive model without the time-consuming feature engineering step. This voided the need to manually extract the characteristics from which the algorithm can learn. Finally, we also explained you how to experimentally validate the resulting model and compare the results of different models.

While the use of deep learning in some cases might eliminate the need for manual feature engineering, it remains important to tune the models with the appropriate parameter settings, which requires the necessary effort. While the presented approach already gives promising results, there are still quite some improvements possible. Therefore, we recommend you to further try tuning the different parameters, and study the effect on the results. You can also continue to experiment with different network architectures, for example by altering the number of layers and nodes.

If you want to gain deeper insights in remaining useful lifetime prediction, a good exercise is trying to cast the prediction problem as a regression or multi-class classification task instead of as a binary classification task. It might also be worth to experiment with the methodology on a larger dataset with a higher number of assets or to try to adapt the different steps to the context of your own dataset.

While the details of each of the steps might differ, the methodological steps we presented are typically the required phases you need to go through when solving a remaining useful lifetime prediction model.

We thank you for completing this video series and hope to welcome you in another AI Starter Kit tutorial.
