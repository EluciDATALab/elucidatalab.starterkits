# EluciDATA Starter Kits

The Starter Kits we have developed generalise and distil the AI needs of specific industrial use cases, which provide a source of inspiration.

Concretely the Starter Kits are self-contained collections of autodidactic material, providing a description of a specific data innovation topic in terms of its business goal, data-related requirements & challenges, relevant data science tasks, etc. These also contain a documented proof-of-concept solution, using public datasets, illustrating which machine learning methods to use and how they should be combined.

The Starter Kits also include relevant links to technology/literature, guidelines and best practices, regulations…
The Starter Kits tackle a wide variety of industrial problem settings, illustrated on specific use cases from different domains. They can be made available to you as a reference solution for data-related innovation problems arising in your specific context.

To guide you in the process of selecting the most appropriate Starter Kit(s) for your company-specific problem setting, each Starter Kit is accompanied by a Starter Kit Passport. Each passport contains:

- a well-defined description of the business context and goal that is dealt with in the Starter Kit,
- the application contexts in which the presented data-driven approach can be useful,
the specific data requirements that need to be satisfied in order to realize such kind of data-driven solution, as well as
- a clear description of the use case we used to illustrate the process.

These passports in itself already provide you with a lot of valuable knowledge on the types of problems you can solve with a data-driven solution, instantiated with examples from a variety of industrial domains.

In addition, each Starter Kit is accompanied by complementary video tutorials to guide you through the different steps AI-based methodology for several industrial use cases, complemented with autodidactic video tutorials. You can access these on our webpage: https://elucidatalab.github.io

Currently, the following Starter Kits are available:
- [SK_1_2_1: Remaining Useful Life Prediction](notebooks/SK_1_2_1_Remaining_Useful_Life_Prediction)
- SK_1_3: Resource Demand Forecasting
- SK_3_1: Advanced Visualisation
- SK_3_2: Data Exploration
- SK_3_4: Time Series Preprocessing

## Running the Starter Kits
Each Starter Kit has its own folder in the notebooks/ directory. In it, there are four files:
- a jupyter file with the notebook (*.ipynb)
- an html rendering of the notebook (*.html)
- a supporting script with data loading and processing functions (support.py)
- a visualizations script with the code to generate the graphics in the notebook (visualizations.py)

In addition, Starter Kit uses code from the starterkits package, which should be installed with:

``` bash
pip install -e .
```

The notebooks have been developed in python 3.7. The required packages with specific versions are listed in requirements.txt.

## The data
The data used in each notebook come from public sources and cover a variety of domains. It is hosted on a different repository (https://github.com/EluciDATALab/elucidatalab.datasets). When you run a notebook for the first time, the raw data will be downloaded from this repo and processed to the format the notebook reads it in. This processed dataset will be saved locally in the data/ folder.
<i class="fa fa-envelope"></i>


<img src="img/PoweredBySirris.png"
    width="25%"
     alt="Powered by Sirris" />

<img src="img/email.png"
    width="5%"
     alt="Powered by Sirris"
     style="float: left; margin-right: 10px;" />
elucidatalab@sirris.be

© Sirris Gebruikersovereenkomst Privacy policy
EluciDATA Lab -The Data and AI Competence Lab of Sirris
