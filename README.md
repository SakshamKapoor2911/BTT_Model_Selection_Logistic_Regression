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
    subgraph Data Preparation
        A[Start] --> B(Load Dataset);
        B --> C(Define Label & Features);
        C --> D(Split Data: Training & Test Sets);
    end

    %% Connections from Data Preparation to the next subgraph
    D --> E(Train Initial LR Model with Default C);

    subgraph Model Training & Evaluation (Default)
        E --> F(Make Predictions: Probability & Class Labels);
        F --> G{Evaluate: Confusion Matrix & Initial Accuracy};
    end

    %% Connections from Model Training & Evaluation (Default) to the next subgraph
    G --> H(Perform GridSearchCV for Optimal C);

    subgraph Model Optimization & Refinement
        H --> I(Train LR Model with Optimal C);
        I --> J(Make Predictions: Probability & Class Labels);
        J --> K{Evaluate: Confusion Matrix & Improved Accuracy};
        K --> L(Visualize: Precision-Recall Curves);
        L --> M(Visualize: ROC Curves & Calculate AUC);
    end

    %% Connections from Model Optimization & Refinement to the next subgraph
    M --> N(Feature Selection: SelectKBest);

    subgraph Feature Engineering & Persistence
        N --> O(Retrain LR Model with Selected Features);
        O --> P{Evaluate: AUC after Feature Selection};
        P -- High AUC? --> Q(Save Optimal Model using Pickle);
        Q --> R[End];
    end

    %% Apply some styling for visual appeal
    classDef start_end fill:#a8e6cf,stroke:#333,stroke-width:2px;
    classDef data_process fill:#ffcccb,stroke:#a33,stroke-width:2px;
    classDef model_eval fill:#d4eeff,stroke:#2b5f7e,stroke-width:2px;
    classDef optimization fill:#fffacd,stroke:#d4a400,stroke-width:2px;
    classDef persist fill:#b8e0d4,stroke:#3a7d44,stroke-width:2px;

    class A,R start_end;
    class B,C,D data_process;
    class E,F,G model_eval;
    class H,I,J,K,L,M optimization;
    class N,O,P,Q persist;

    %% Dotted lines for specific transitions
    style H stroke-dasharray: 5 5;
    style N stroke-dasharray: 5 5;

    %% Edge labels for clarity
    E -- Predict --> F;
    I -- Predict --> J;
    O -- Predict --> P;
