# Superhost Prediction with Logistic Regression

This repository contains the code and resources for a machine learning project that predicts whether an Airbnb host is a 'superhost'. The project utilizes a Logistic Regression model, with hyperparameter tuning performed using GridSearchCV to optimize performance.

## Project Structure

This project includes the following key components:

* **`airbnbData_train.csv`**: The dataset used for training and evaluating the model. This dataset has undergone preprocessing, including one-hot encoding, scaling, and imputation of missing values.
* **`best_logistic_regression_model.pkl`**: A pre-trained Logistic Regression model saved using Python's `pickle` module. This allows for easy deployment and use of the best-performing model without retraining.

## Machine Learning Workflow

The following diagram illustrates the general machine learning workflow followed in this project:

```mermaid
graph TD
    A[Start] --> B(Load Data);
    B --> C{Define Label & Features};
    C --> D[Split Data: Training & Test Sets];
    D --> E(Train Initial LR Model with Default C);
    E --> F{Evaluate Initial Model: Confusion Matrix};
    F --> G(Perform GridSearchCV for Optimal C);
    G --> H(Train Optimal LR Model);
    H --> I{Evaluate Optimal Model: Confusion Matrix};
    I --> J(Plot Precision-Recall Curves);
    J --> K(Plot ROC Curves & Calculate AUC);
    K --> L(Perform Feature Selection - SelectKBest);
    L --> M(Retrain & Evaluate with Selected Features);
    M --> N(Save Optimal Model using Pickle);
    N --> O[End];
