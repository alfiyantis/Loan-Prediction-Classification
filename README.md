Loan-Prediction-Classification
===

## Table of Contents
* **[Background and Objective](#backg)**
* **[Scope of Problem](#scope)**
* **[Data and Assumptions](#datas)**
* **[Data Analysis](#analys)**
* **[Summary](#summary)**
* **[Suggestion](#suggest)**

<a name="backg"></a>
## Background and Objective
A banking company offers money loans to its customers. However, 12.3% of customers who applied for a loan failed to repay the loan. It will affect the financial health of the company. If not handled immediately, the percentage of defaulting customers will increase so that the company can lose money and cause new problems for the company.
Then the manufacture of machine learning is needed to predict customers who have the possibility of default so that companies can avoid providing loans to prospective customers who have the possibility to be default.

<a name="scope"></a>
## Scope of Problem
The problem to be solved is to reduce the percentage of defaulting customers.
The output to be produced is the prediction of customers who have the possibility of default so that the company can reduce the percentage of default customers.

<a name="datas"></a>
## Data and Assumptions
The data used is from India.
The dataset consists of 12 features:
1. Id = ID of the user
2. Income = Income of the user
3. Age = Age of the user
4. Experience = Professional experience of the user in years
5. Married/Single = Marital status of the user
6. House_Ownership = Residence ownership status of the user
7. Car_Ownership = Car ownership status of the user
8. Profession = Profession of the user
9. CITY = City of residence
10. STATE = State of residence
11. CURRENT_JOB_YRS = Years of experience in the current job
12. CURRENT_HOUSE_YRS = Number of years in the current residence
13. Risk_Flag (target) = Defaulted on a loan

The dataset consists of 12 columns and 252000 rows
Only 9 features will be used. Some features will be dropped, such as `ID`, `CURRENT_JOB_YRS`, and `CITY`.

[Data imbalance between  default and non-default customers]

![imbalance data](https://user-images.githubusercontent.com/105115689/168623853-c5138698-9ed8-4a43-8735-f36395bb3793.png)

[Correlation between numerical features]

![correlation](https://user-images.githubusercontent.com/105115689/168577539-39bf0cde-49f2-4be8-87a5-548d65370cf1.png)

-	The ID feature is dropped because they all contain unique values
-	The CURRENT_JOB_YRS feature is dropped because it has a strong correlation with the experience feature, so only one of the two features is needed to be used in making machine learning (the photo above)
-	The CITY feature is dropped because there are many unique values

<a name="analys"></a>
## Data Analysis
the data is normally distributed and there are no outliers in the data used (can be seen in the image below
[Numerical features distribution with boxplot]

![boxplot nums](https://user-images.githubusercontent.com/105115689/168577698-1f78449e-07a7-435d-8dd1-0f1ffd13ebcf.png)

[Numerical features distribution with kdeplot]

![kdeplot nums](https://user-images.githubusercontent.com/105115689/168577719-890ed370-eecb-4841-b010-1258028b7c13.png)

The problem to be solved is to find out what kind of customers fail to pay based on the available data, along with the distribution of customers who fail to pay according to some features.
1.	Distribution of risk customers by age

![Distribusi risk customers berdasarkan usia](https://user-images.githubusercontent.com/105115689/168577868-35b6d97a-a9e7-4f20-b5b1-63e550f6e471.png)

2.	Distribution of risk customers by house

![Distribusi risk customers berdasarkan house ownership](https://user-images.githubusercontent.com/105115689/168577918-161af82d-7929-408e-b0ae-de7eb8c95110.png)

3.	Distribution of risk customers by car

![Distribusi risk customers berdasarkan car ownership](https://user-images.githubusercontent.com/105115689/168577972-b5730f94-8e4e-418c-b063-4a19d3a4b4b8.png)

4.	Distribution of risk customers by marital status

![Distribusi risk customers berdasarkan marital status](https://user-images.githubusercontent.com/105115689/168578187-b22ff701-0ab1-4f22-8637-a21f9ca6f899.png)

5.	Distribution of risk customers by profession

![Distribusi risk customers berdasarkan profesi](https://user-images.githubusercontent.com/105115689/168578270-bebd01af-6780-4c23-8f27-45f70f0a7063.png)

6.	Distribution of risk customers by state

![Distribusi risk customers berdasarkan state](https://user-images.githubusercontent.com/105115689/168578303-1a679d3b-7361-41b5-af4d-8d8c1ac810fa.png)

From the bar chart above, it can be concluded that the default customers are mostly young adults, who live in rental houses, do not own a car, work as police officers, and come from Manipur.

Categorical features with string data types are converted into integer data types with label encoding techniques (married/single), one hot encoding (own house), and sorting based on the data distribution (state rank and profession rank).

There are no outliers and null values in the data, but there is a class imbalance in the data. The SMOTE technique `(0.4)` is used to overcome the class imbalance, no undersampling technique is used so that no data is discarded.

Not too many features are available, so almost all of the features are used in the creation of this machine learning.
There are 11 features used:
1. Income
2. Age
3. Experience
4. Married/Single
5. Car_ownership
6. CURRENT_HOUSE_YRS
7. Ownhouse_norent_noown
8. Ownhouse_owned
9. Ownhouse_rented
10. State_rank
11. Profession_rank

Three different models are used for the creation of this machine learning, the model with the best score will be used.
1. K-Nearest Neighbors (KNN)
2. Decision Trees
3. Random Forest

The main purpose of making this machine learning is to reduce default customers, so it is necessary to focus on the score of accuracy, precision, recall, and AUC. it aims to reduce the number of false-positives and false-negatives
[Score of each model in percentage (%)]

| | K-Nearest Neighbors | Decision Trees | Random Forest |
|---|---|---|---|
| Accuracy | 87.66 | 88.67 | 90.72 |
| Precision | 75.27 | 74.90 | 80.16 |
| Recall | 84.09 | 90.24 | 89.33 |
| AUC | 86.58 | 89.14 | 90.30 |

Of the 3 models that have been tested, the random forest model has the highest score, then do the hyperparameter tuning to maximize the performance of the model. All scores increases after hyperparameter tuning process is executed.
| | Before | After | Increase |
|---|---|---|---|
| Accuracy | 90.72 | 90.90 | 0.18 |
| Precision | 80.16 | 80.54 | 0.38 |
| Recall | 89.33 | 89.52 | 0.19 |
| AUC | 90.30 | 90.48 | 0.18 |


![feature importance](https://user-images.githubusercontent.com/105115689/168590353-bcf40d64-8253-4bdd-852b-07f7e078cc68.png)

5 features have lowest score when compared to the other 6 features, so the 5 features can be dropped:
1. Car_Ownership
2. Married/Single
3. Ownhouse_norent_noown
4. Ownhouse_owned
5. Ownhouse_rented

Almost all scores decreased after dropping the 5 features that had the lowest score, except for the recall score. The score reduction is not significant, besides the process of reducing less important features can also save time in running code. By using the features and the final model (random forest with 6 features), the model is considered quite good and does not take too long because fewer features are used.
| | Before | After | Differences |
|---|---|---|---|
| Accuracy | 90.90 | 90.88 | 0.02 |
| Precision | 80.54 | 80.48 | 0.06 |
| Recall | 89.52 | 89.53 | 0.01 |
| AUC | 90.48 | 90.47 | 0.01 |

<a name="summary"></a>
## Summary
From the modeling results above, it can be concluded that the best prediction results come from a combination of the random forest model with 6 important features:
1. Income
2. Age
3. Profession_rank
4. Experience
5. State_rank
6. CURRENT_HOUSE_YRS

The most important feature with the highest score is income. meaning that income is the most relevant feature in predicting default customers.

<a name="suggest"></a>
## Suggestion
* The company can classify customers based on salary (considering salary is the most important feature), so it has its target market in each customer group to reduce default customers.
* The company need to add more customer financial information to get more features so that it can develop to a better model
the company can tighten lending for customers from Manipur. The second option is for the company make a special offer for customers from Manipur so that customers can still borrow money under special conditions.
