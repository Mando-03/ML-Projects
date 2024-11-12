
---

# House Price Prediction 🏡

This project aims to predict house prices using various machine learning models. It implements several regression techniques and includes a thorough process of data preprocessing, feature engineering, model building, evaluation, and hyperparameter tuning to achieve optimal results.

## Project Overview

The goal of this project is to predict house prices based on the Ames Housing Dataset. The project goes through the entire process of data cleaning, feature engineering, and implementing various machine learning models, including ensemble techniques like Voting and Stacking regressors, to improve prediction accuracy.

## Table of Contents
 
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Saving and Deployment](#saving-and-deployment)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Mando-03/ML-Projects
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized as follows:

- **Data/**: Contains the training and test datasets (`train.csv`, `test.csv`).
- **Models/**: Contains saved models like Voting and Stacking regressors.
- **Notebooks/**: Contains Jupyter notebooks used for exploratory data analysis and modeling.
- **Savings/**: Contains saved preprocessing encoders and the scaler for deployment.
- **main.py**: The main Python script for running the model.
- **requirements.txt**: Lists all dependencies for running the project.

## Feature Engineering

We performed extensive feature engineering, including:

1. **Dropping Irrelevant Columns**: Removed columns with low variance, high correlation, or too many null values.
2. **Handling Missing Values**: Imputed missing values using mean for numeric features and mode for categorical ones.
3. **Creating New Features**: Introduced new features like `HouseAge` and `YearsSinceRemodAdd`.
4. **Encoding Categorical Features**:
   - **Ordinal Encoding**: For columns where the order of categories matters (e.g., `ExterQual`, `KitchenQual`).
   - **One-Hot Encoding**: For low cardinality categorical features (e.g., `MSZoning`, `Neighborhood`).
   - **Target Encoding**: For high-cardinality features (e.g., `SaleCondition`, `Exterior1st`).

## Models Used

The following models were used:

1. **Lasso & Ridge Regression**: Useful for regularization and handling multicollinearity.
2. **Random Forest**: An ensemble method reducing overfitting by averaging trees.
3. **Gradient Boosting**: Combines weak learners for strong performance.
4. **XGBoost**: Efficient gradient boosting implementation with excellent performance.
5. **Extra Trees**: Similar to Random Forest but faster and with more randomness.
6. **Voting Regressor**: Combines the predictions of multiple models for improved accuracy.
7. **Stacking Regressor**: Combines base models and uses a final estimator to make predictions.

--

## Evaluation Metrics

The models were evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **Mean Squared Error (MSE)**: Squared difference between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretability.
- **R² Score**: The proportion of variance explained by the model.

## Results

After hyperparameter tuning and model ensembling, the Voting Regressor yielded the best results .

## Saving and Deployment

The final model and preprocessing steps were saved for deployment:

- **Encoders**: Ordinal, target, and one-hot encoders were saved for transforming input data.
- **Scaler**: A `StandardScaler` was saved for scaling numeric features.
- **Voting Regressor Model**: The final trained model was saved using `joblib`.

To load and use the model:

```python
import joblib
model = joblib.load('Savings/voting_regressor_model.pkl')
```

---

## MLFlow Integration

To improve experiment tracking and model reproducibility, we have integrated **MLFlow** to systematically track model parameters, metrics, and logs. Below is the summary of MLFlow implementation:

1. **Setup MLFlow Experiment**: We initialize an MLFlow experiment for tracking all model runs.

2. **Model Training and Logging**: The models are trained and their parameters, metrics (such as MAE, MSE, RMSE, R²), and models are logged to MLFlow.

3. **Saving Models in MLFlow**: Models are saved in the MLFlow tracking server, ensuring that each experiment is logged and easily accessible.

This will allow you to track multiple experiments with different models and configurations. By using MLFlow, you'll be able to keep track of all the hyperparameters, metrics, and the final model that performs best, ensuring reproducibility and easy comparison of results across different model configurations.

