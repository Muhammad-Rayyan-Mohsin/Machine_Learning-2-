# Machine_Learning-2-


# Fraud Detection Model

This project focuses on building a fraud detection model using machine learning techniques. The dataset used contains various features related to individuals' attributes and their taxable income. We aim to predict whether an individual's taxable income exceeds $30,000, which serves as a binary classification problem (fraudulent vs. non-fraudulent).

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Conclusion](#conclusion)

## Introduction

Fraud detection is a critical task in financial and banking sectors to identify potential fraudulent activities and prevent financial losses. Machine learning models can analyze patterns and anomalies in data to predict fraudulent behavior. This project aims to develop a fraud detection model using a decision tree classifier and evaluate its performance.

## Data Preprocessing

The dataset used in this project required preprocessing to prepare it for modeling. We performed the following steps:

- Encoding categorical variables using one-hot encoding.
- Scaling numerical features using MinMaxScaler to bring them to a common scale.
- Defining the target variable as a binary classification problem by categorizing taxable income exceeding $30,000 as fraudulent (1) and below as non-fraudulent (0).
- Saving the preprocessed data to a new CSV file for further analysis.

## Model Training

We split the preprocessed data into training and test sets, with 421 samples for training and 180 samples for testing. We trained two models:

1. **Linear Regression Model**: Used for baseline prediction of taxable income.
2. **Decision Tree Classifier**: Employed for binary classification of fraudulent and non-fraudulent cases based on individuals' attributes.

## Model Evaluation

We evaluated the performance of the decision tree classifier using classification metrics such as precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify fraudulent and non-fraudulent cases. The classification report summarizes the model's performance on the test set.

## Conclusion

This project demonstrates the application of machine learning techniques in fraud detection, a critical task in various industries. By preprocessing data, training a decision tree classifier, and evaluating its performance, we aim to provide insights and tools for detecting fraudulent activities effectively.
