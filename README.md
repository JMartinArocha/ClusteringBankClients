# ClusteringBankClients

![ClusteringBankClients](ClusteringBankClients.png)


This repository contains a practical exercise focused on clustering techniques, designed to train and enhance skills in data analysis and machine learning. The context of this exercise revolves around a dataset from direct marketing campaigns via phone calls conducted by a Portuguese banking institution. The primary goal of this classification project is to predict whether a client will subscribe to a term deposit, which is marked by the target variable 'y'.

The aim of this project is to apply various clustering algorithms to understand better and segment the bank's clientele based on their characteristics and their responses to previous marketing campaigns. By exploring different clustering approaches such as K-means, DBSCAN, and Density Peak Clustering (DPC), this exercise seeks to identify distinct customer groups, which can help in devising more tailored marketing strategies that could potentially increase the subscription rates of term deposits.

Github: https://github.com/JMartinArocha/ClusteringBankClients

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Badges

![GitHub license](https://img.shields.io/github/license/JMartinArocha/ClusteringBankClients.svg)
![Python version](https://img.shields.io/badge/python-3.x-blue.svg)
![last-commit](https://img.shields.io/github/last-commit/JMartinArocha/ClusteringBankClients)
![issues](https://img.shields.io/github/issues/JMartinArocha/ClusteringBankClients)
![commit-activity](https://img.shields.io/github/commit-activity/m/JMartinArocha/ClusteringBankClients)
![repo-size](https://img.shields.io/github/repo-size/JMartinArocha/ClusteringBankClients)

### About virtual envs

Using a virtual environment in Python is a best practice for managing project dependencies. It allows you to create a self-contained directory that contains all the Python executable files and packages you need for your project. This way, you can avoid conflicts between package versions across different projects.

Why Use a Virtual Environment?
Dependency Management: Each project can have its dependencies without affecting other projects or the global Python installation.
Reproducibility: Makes it easier to share and collaborate with others, ensuring that everyone has the same setup.
Isolation: Prevents inadvertent changes to system files and configurations.
Setting Up a Virtual Environment
Here’s how to set up a virtual environment for your clustering project:

1. Install Virtual Environment
First, you need to install the virtualenv package if it's not already installed. This package allows you to create virtual environments in Python. You can install it using pip:

```shell
pip install virtualenv
```

2. Create a Virtual Environment
Navigate to your project directory, or where you want to set up your project:

```shell
cd path/to/your/project
```

Now, create a virtual environment within this directory:

Here, venv is the name of the virtual environment directory. You can name it anything, but venv or .venv is typical.

``` shell
virtualenv venv
```

3. Activate the Virtual Environment
Before using the environment, you need to activate it:

```shell
venv\Scripts\activate # Windows
source venv/bin/activate # Linux/macOS
```

Deactivate the Environment
To stop using the virtual environment and go back to your global Python, you simply type:

```shell
deactivate
```

Keeping Track of Dependencies
To make it easier for others to set up the same environment, it’s good practice to save your dependencies in a requirements.txt file:

```shell
pip freeze > requirements.txt
```

This command writes a list of all installed packages and their versions to the requirements.txt file. Others can install all required packages using:

```shell
pip install -r requirements.txt
```

### Prerequisites

Before running the scripts, it's essential to install the necessary Python packages. The project has a `requirements.txt` file listing all the dependencies. You can install these using `pip`. 

Note: The commands below are intended to be run in a Jupyter notebook environment, where the `!` prefix executes shell commands. If you're setting up the project in a different environment, you may omit the `!` and run the commands directly in your terminal.

```bash
!pip3 install --upgrade pip
!pip3 install -r requirements.txt
```

## Importing Shared Utilities

The project utilizes a shared Python utility script hosted on GitHub Gist. This script, `ml_utilities.py`, contains common functions and helpers used across various parts of the project. To ensure you have the latest version of this utility script, the project includes a step to download it directly from GitHub Gist before importing and using its functions.

Below is the procedure to fetch and save the `ml_utilities.py` script programmatically:

## Data clean and normalization

This section of the code involves the preprocessing of categorical data within the dataset. We utilize the LabelEncoder from sklearn.preprocessing to convert each categorical column into a format that can be easily used by machine learning algorithms. This step is crucial as it translates categorical labels into a numeric format where each unique label in a column is assigned a corresponding integer. The included columns such as 'land_surface_condition', 'foundation_type', among others, are transformed to enhance the model's ability to learn from these features effectively.

Missing values are filled using the forward fill method to maintain data continuity. The dataset is then normalized using a custom utility function with MinMaxScaler, adjusting feature scales to enhance model performance. Furthermore, the dataset is merged with damage_grade labels for direct analysis. Data type conversions ensure consistent handling of features. To address potential class imbalance, the dataset is resampled to equalize the number of samples across different damage grades. This balancing act is visualized in a bar chart, showcasing the distribution of buildings across damage grades post-resampling.


# Feature selection

Given the objective of predicting whether a client will subscribe to a term deposit based on the provided data, we need to carefully select features that might influence this decision. Let's analyze the potential impact of each feature in the context of predicting the likelihood of a subscription.

- Age (numeric): Different age groups might have different financial needs and priorities, affecting their likelihood to invest in term deposits.

- Job (categorical): Certain professions may have more disposable income or different financial planning tendencies, influencing term deposit subscriptions.

- Marital (categorical): Marital status can affect financial stability and decisions regarding savings or investments.

- Education (categorical): Education level often correlates with financial literacy, which could influence investment decisions like term deposits.

- Default (binary): If a client has defaulted before, they might be seen as a higher risk and could be less likely to be approved for or interested in further financial products.

- Balance (numeric): Higher balances might indicate a higher likelihood of investing part of it in term deposits.

- Housing (binary): Owning a home or having a housing loan could influence financial stability and investment decisions.

- Loan (binary): Having personal loans might affect a client's financial capacity to commit to a term deposit.

- Contact (categorical): The method of communication might impact the effectiveness of the marketing campaign.

- Day (numeric): The day of the last contact could reflect on the urgency or receptiveness to the offer.

- Month (categorical): Seasonal factors or financial cycles throughout the year could influence the decision to invest in term deposits.

- Duration (numeric): Longer call durations might indicate higher interest or a more successful persuasion by the marketer.

- Campaign (numeric): The number of contacts could either lead to higher engagement or cause annoyance, impacting the decision.

- Pdays (numeric): The time since the last campaign could affect the freshness of the client's engagement and their decision-making.

- Previous (numeric): Frequent contacts in past campaigns could indicate interest or annoyance, influencing current decisions.

- Poutcome (categorical): The outcome of previous marketing campaigns can significantly influence current decisions based on past satisfaction or dissatisfaction.

# PCA

PCA reduces the dimensionality of the data by transforming the original variables into a new set of variables (principal components) that are uncorrelated and ordered so that the first few retain most of the variation present in all of the original variables. This helps in simplifying the model without significant loss of information.


Interpretation of the PCA Results:
Principal Component 1 (PC-1) and Principal Component 2 (PC-2) represent combinations of your original variables. The coefficients (loading scores) for each feature tell us how much each feature contributes to each principal component. Here’s how to interpret these results:

High Loadings on PC-1:

pdays (0.654542)
previous (0.619908)
poutcome_unknown (-0.251315)
poutcome_failure (0.157970)
contact_unknown (-0.095359)
campaign (-0.187126)

These features have the strongest weights in PC-1, suggesting that aspects like the number of days since the client was last contacted and the number of contacts in previous campaigns are crucial in explaining the variability in your dataset along PC-1. The negative signs indicate an inverse relationship with the principal component axis.

High Loadings on PC-2:

age (0.790916)
balance (0.461500)
housing_no (0.145966)
marital_single (-0.167218)
month_may (-0.093303)

These features contribute significantly to PC-2. The age and account balance of clients, whether they have housing loans, their marital status, and the timing of contact (month of May) play significant roles. The direction (positive or negative) of these loadings indicates how each feature correlates with this component.

### How These Insights Help Determine Feature Importance:

Direction of Influence: Positive and negative values indicate the direction of the relationship with the principal component. For example, features like pdays and previous have a positive correlation with PC-1, suggesting that higher values of these features align with higher values on PC-1.

Magnitude of Influence: The magnitude of the loading scores indicates the strength of the influence. Larger absolute values mean a stronger influence. For example, age and balance have substantial loadings on PC-2, highlighting their strong influence on the dataset's variance along this axis.

### Strategic Decisions for Feature Selection:

Focus on High Loadings: Features with higher absolute values in the loading scores are typically more influential in explaining variance and potentially more useful for predictive modeling. For example, focusing on pdays, previous, age, and balance might yield a model that captures significant variability in client behavior regarding term deposit subscriptions.

Reduce Dimensionality: PCA helps in reducing the number of features by creating new components based on the original features that explain most of the data's variability. You might choose to use only the first few principal components as inputs to your predictive model if they capture a sufficient percentage of the total variance.

Consider Multicollinearity: Since PCA components are orthogonal (independent), using them instead of correlated original features can help in models where multicollinearity is a concern, such as linear regression.

### PCA conclusion:

From the PCA analysis, we can conclude that features like the number of days since last contact, the number of contacts before the current campaign, client's age, and balance are particularly significant in explaining the variability in how clients react to term deposit offers. These insights can guide further feature engineering and model development, helping focus on the most informative variables for predicting term deposit subscriptions.

## Silhouette Score

The silhouette score is a metric used to evaluate the quality of clusters created by a clustering algorithm. It measures how similar an object is to its own cluster compared to other clusters. The silhouette score for each sample is a value between -1 and +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

### How to Interpret the Silhouette Score

Values close to +1 indicate that the sample is far away from the neighboring clusters.
A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters.

Values less than 0 generally indicate that those samples might have been assigned to the wrong cluster.

High average silhouette scores across all samples indicate robust and distinct clustering. If the scores are low or negative, it may suggest overlapping clusters, which could indicate that the number of clusters (K) is set improperly or the clustering algorithm does not fit the dataset well.

## Dendrogram Analysis

A dendrogram is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering. It provides a visual insight into the structure of the data, showing the hierarchical relationship between clusters at different levels of granularity. Each merge is represented by a horizontal line joining clusters, with the height of the line representing the distance or dissimilarity between these clusters.

Dendrograms are particularly useful in helping to decide the number of clusters by visually indicating where clusters can be "cut" based on the height of the joins. This visualization is essential for understanding the clustering dynamics and can guide the selection of an appropriate threshold for meaningful cluster separation.

