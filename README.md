# deep-learning-challenge

# Alphabet Soup Deep Learning Report

## Overview of the Analysis

The goal of this analysis was to predict which applicants would be successful if funded by Alphabet Soup. If we can successfuly build a model using historical funding and success data from Alphabet Soup's past ventures, then we can more confidently make determinations on funding in the future. We built a deep learning model using tenserflow and predicted the success of firms with 73% accuracy. Below is a summary of the steps taken to build this model, and more information about the results.

## Results

### Data Preprocessing

* The target variable for our model is `IS_SUCCESSFUL`. This variable denotes whether a firm funding by Alphabet Soup was ultimately successful or not.
* Our model used many of the input variables as features: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
* We did not use the variables `EIN` and `NAME` as features since these are unique to each record and would not be beneficial to training the model.

### Compiling, Training, and Evaluating the Model

We looked at the balanced accuracy score, confusion matrix, and precision and recall scores to judge the effectiveness of the model. These were calculated by predicting the loan status of loans in the test set and comparing those predictions to the actual observations.

* We used keras_tuner to find the best neural network within a given space. We ran the tuner multiple times, leaving out certain variables each time and allowing more or less epochs. Eventually, the best model found was one with 6 layers, with [45, 13, 33, 17, 75, 31] neurons in each layer respectively and 73 neurons in the input layer.
* To increase model performance, we removed redundant dummy variables that were fully explained by other variables, such as `AFFILIATION_Other` or `CLASSIFICATION_Other`. We also removed `STATUS` as there were only six non-1 records for that feature.
* Unfortunately, we couldn't achieve accuracy over 75% even after multiple runs through the tuner with more epochs on each test. Ultimately it seems very hard to predict which firms will be successful, and we have to accept that this model can't fully explain the probability of success using the provided variables.

## Summary
We can predict which firms will be successful with 73% accuracy. The final model found in the optimization process is better than the simple model used in the first notebook, but only marginally. Both round to 73% accuracy. Given this fact, a simple model like a logistic regression would probably do almost as well, and would be much quicker to train and easier to work with. We recommend exploring this avenue if time permits, and we should also consider collecting more data about from each firm to see if there are variables with more explanatory power that could help a model achieve >75% accuracy.
