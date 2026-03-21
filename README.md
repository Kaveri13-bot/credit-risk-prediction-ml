# Credit Risk Prediction using Machine Learning
**Project Overview**
This project builds a Machine Learning model to predict whether a customer is likely to default on a loan. Financial institutions can use this model to assess credit risk and make better lending decisions.
The model analyzes customer financial and demographic data to classify loan applicants into low-risk and high-risk categories.

**Problem Statement**
Banks and financial institutions face significant losses when borrowers default on loans.
This project aims to develop a predictive model that identifies risky borrowers before a loan is approved.

# Dataset Description
The dataset contains customer financial information such as:
Age
Income
Employment status
Loan amount
Credit history
Debt-to-income ratio
Loan purpose

**Machine Learning Model used:**
This project uses the XGBoost (Extreme Gradient Boosting) algorithm to predict credit risk.
XGBoost is a powerful ensemble learning technique based on gradient boosting that is widely used in financial risk modeling due to its high performance and ability to handle structured/tabular data.

# Key advantages of using XGBoost in this project
• Handles missing values effectively

• Captures complex feature interactions

• Reduces overfitting through regularization

• Provides high predictive accuracy

The model was trained using customer financial and demographic data to classify whether a borrower is likely to default on a loan.
Target Variable:

0 → Non-Default

1 → Default

# Model Performance
The XGBoost model was evaluated using standard classification metrics.
Evaluation Metrics:
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

# Model Evaluation
The model performance on the validation dataset is shown below:

Class	             Precision	Recall	F1-Score	Support

0 (Loan Rejected)	  0.68    	0.57	   0.62	     30

1 (Loan Approved)	  0.82	    0.88	   0.85	     69

Overall Performance

Metric	              Score

Accuracy	            0.79

Macro Average F1	    0.74

Weighted Average F1	  0.78

The results indicate that the XGBoost model performs well in identifying high-risk borrowers.

# Conclusion
•	Built a classification model to predict credit risk using real-world dataset

•	Performed data preprocessing, feature engineering, and model tuning

•	Achieved 79% accuracy with balanced precision and recall

•	Evaluated performance using confusion matrix and classification report

# Technologies Used
Python

Pandas

NumPy

Scikit-learn

Matplotlib

SQL (for data analysis)

# Project Workflow
Data Collection

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Model Training

Model Evaluation

Prediction


