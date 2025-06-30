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
    %% Node Definitions (flat list first)
    A[Start]
    B(Load Dataset)
    C(Define Label & Features)
    D(Split Data: Training & Test Sets)
    E(Train Initial LR Model with Default C)
    F(Make Predictions: Probability & Class Labels)
    G{Evaluate: Confusion Matrix & Initial Accuracy}
    H(Perform GridSearchCV for Optimal C)
    I(Train LR Model with Optimal C)
    J(Make Predictions: Probability & Class Labels)
    K{Evaluate: Confusion Matrix & Improved Accuracy}
    L(Visualize: Precision-Recall Curves)
    M(Visualize: ROC Curves & Calculate AUC)
    N(Feature Selection: SelectKBest)
    O(Retrain LR Model with Selected Features)
    P{Evaluate: AUC after Feature Selection}
    Q(Save Optimal Model using Pickle)
    R[End]

    %% Connections
    A --> B
    B --> C
    C --> D
    D -- Data Processed --> E
    E --> F
    F --> G
    G -- Optimization Needed --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M -- Feature Analysis --> N
    N --> O
    O --> P
    P -- High AUC? --> Q
    Q --> R

    %% Subgraph Grouping (applies visual grouping to already defined nodes)
    subgraph Data Preparation
        A --> D
    end

    subgraph Model Training & Evaluation (Default)
        E --> G
    end

    subgraph Model Optimization & Refinement
        H --> M
    end

    subgraph Feature Engineering & Persistence
        N --> R
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

    %% Edge labels (already defined in connections section)
    %% Note: Mermaid will apply these from the earlier definition.
    %%       No need to redefine them here if they are simple text labels.
