# Titanic Survival Prediction Model
## Overview
This repository contains a machine learning model designed to predict the survival of passengers aboard the Titanic based on several features. The dataset used for training and testing the model was sourced from Kaggle and consists of two .csv files: train.csv and test.csv.

## Dataset Information
train.csv: This file contains the training dataset with information about passengers including:
- Name
- PassengerID
- Survived (Target Label)
- Age
- Sex
- Pclass (Ticket class)
- SibSip (Number of siblings/spouses aboard)
... (and other related features)
  
test.csv: This file contains the test dataset with the same features as train.csv, except for the Survived column, which is the target label that needs to be predicted.

## Model Selection and Evaluation
Several machine learning classification algorithms were implemented and evaluated to determine the best-performing model for this task. The following algorithms were considered:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- AdaBoosting Classifier
- Voting Classifier
  
## Evaluation Metrics
The accuracy score was used as the primary metric to evaluate the performance of each model. Cross-validation techniques were employed to ensure the reliability of the results.

## Model Deployment
After thorough evaluation and hyperparameter tuning, the Random Forest Classifier was chosen as the final model due to its superior performance on both the training and validation datasets.

## Prediction and Results
The trained Random Forest model was applied to the test dataset (test.csv) to predict the survival outcomes for the passengers. The predictions were then saved in an .csv file, which includes the PassengerID and the predicted Survived columns.

## Requirements
- Python (>= 3.6)
- NumPy
- pandas
- scikit-learn
- XGBoost
