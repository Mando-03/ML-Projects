
---

# Smart Credit Card Fraud Detection System

![Spam Img](https://revnew.com/hubfs/Revnew/Images/content/how-to-avoid-hitting-spam-lists-10-powerful-email-marketing-tips-for-2023.jpg)

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. It tackles the challenge of **class imbalance** (fraud vs. non-fraud) by implementing several strategies such as **SMOTEENN resampling**, **anomaly detection**, and **ensemble models** like **XGBoost**, **Random Forest**, and **Isolation Forest**.

## Table of Contents
- [EDA](#eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Cross-Validation](#cross-validation)
- [Best Model](#best-model)
- [Saving the Model](#saving-the-model)



## EDA

- Initial data exploration revealed significant class imbalance between fraudulent and non-fraudulent transactions.
- We dropped duplicates and visualized transaction **amount**, **time**, and **correlations** between features.

Key visualizations include:
- **Transaction amount** and **time distributions**
- **Correlation heatmaps**
- **Class balance count plot**


### Handling Class Imbalance
We applied **SMOTEENN** to handle the class imbalance:
```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, Y)
```

### Data Scaling
We used **StandardScaler** for scaling the features:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Modeling

We employed various models, including anomaly detection models like **Isolation Forest** and ensemble methods like **Random Forest**, **XGBoost**, and **AdaBoost**:


## Evaluation

We evaluated each model on the following metrics:
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**

### Confusion Matrix and Heatmap

## Cross-Validation

## Best Model

After evaluating all models, **XGBoost** emerged as the best performer in terms of **accuracy** and **F1-Score**.


## Saving the Model

## Conclusion

This project highlights the importance of handling class imbalance in fraud detection systems. **XGBoost** proved to be the best model for this problem, but models like **Random Forest** and **Gradient Boosting** also performed well.

---

### Key Resources:
- **Dataset**: [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

--- 
