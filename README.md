# Supervised-Learning
# Logistic Regression - Pass/Fail Prediction

## 📌 Project Overview
This project demonstrates the basic implementation of **Logistic Regression** using **Python and Scikit-learn** to predict whether a student will **Pass or Fail** based on the number of hours studied.

Logistic Regression is a **supervised machine learning algorithm** commonly used for **binary classification problems**.

In this project:
- Input: Hours studied
- Output: Pass (1) or Fail (0)

---

## 🧠 Algorithm Used
**Logistic Regression**

Logistic Regression predicts probabilities using the **sigmoid function**, which converts linear outputs into values between **0 and 1**.

\[
P(Y=1) = \frac{1}{1 + e^{-z}}
\]

Where:
- \(z = b0 + b1X\)

---

## 📂 Project Structure
```
Logistic-Regression-Project
│
├── Logistic_regression.ipynb   # Jupyter notebook implementation
└── README.md                   # Project documentation
```

---

## ⚙️ Technologies Used
- Python
- NumPy
- Scikit-learn
- Jupyter Notebook

---

## 📊 Dataset
A simple sample dataset is used:

| Hours Studied | Result |
|---------------|--------|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Pass |
| 5 | Pass |

Where:
- **0 = Fail**
- **1 = Pass**

---

## 🚀 Implementation Steps

1. Import required libraries
2. Create sample dataset
3. Initialize Logistic Regression model
4. Train the model using `model.fit()`
5. Predict results using `model.predict()`
6. Calculate probability using `model.predict_proba()`

---

## 💻 Code Example

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X, y)

# Prediction
prediction = model.predict([[3.5]])
probability = model.predict_proba([[3.5]])

print("Prediction (0=Fail, 1=Pass):", prediction)
print("Probability:", probability)
```

---

## 📈 Example Output

```
Prediction (0=Fail, 1=Pass): [1]
Probability: [[0.41 0.59]]
```

This means the model predicts the student will **Pass** with about **59% probability**.

---

## 🎯 Learning Outcome
Through this project you will learn:

- Basics of **Logistic Regression**
- Binary classification in **Machine Learning**
- Model training and prediction
- Using **Scikit-learn** for ML tasks

---

# Decision Tree Machine Learning Project

This repository demonstrates the implementation of **Decision Tree algorithms** using **Python and Scikit-learn**.
The project includes two main machine learning tasks:

* **Decision Tree Classification**
* **Decision Tree Regression**

Both implementations are provided in Jupyter notebooks to help understand how decision trees work for different types of prediction problems.

---

# Project Files

```
Decision_tree_classifier.ipynb
Decision_tree_regressor.ipynb
README.md
```

---

# Decision Tree Overview

A **Decision Tree** is a supervised machine learning algorithm used for both **classification and regression tasks**.
It works by splitting the dataset into smaller subsets based on feature values, creating a structure that resembles a tree.

### Key Characteristics

* Easy to understand and interpret
* Works for both categorical and numerical data
* Requires minimal data preprocessing
* Useful for decision-making and prediction tasks

---

# Decision Tree Classification

Decision Tree Classification is used when the **target variable is categorical**.

### Example Problem

Predict whether a student **passes or fails** based on **hours studied**.

### Sample Dataset

| Hours Studied | Result |
| ------------- | ------ |
| 1             | Fail   |
| 2             | Fail   |
| 3             | Fail   |
| 4             | Pass   |
| 5             | Pass   |

### Example Code

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])  # 0 = Fail, 1 = Pass

# Model
model = DecisionTreeClassifier()

# Train
model.fit(X, y)

# Prediction
prediction = model.predict([[3.5]])

print("Prediction (0 = Fail, 1 = Pass):", prediction)
```

---

# Decision Tree Regression

Decision Tree Regression is used when the **target variable is numerical or continuous**.

### Example Problem

Predict **salary** based on **years of experience**.

### Sample Dataset

| Experience | Salary |
| ---------- | ------ |
| 1          | 10000  |
| 2          | 20000  |
| 3          | 30000  |
| 4          | 40000  |
| 5          | 50000  |

### Example Code

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10000, 20000, 30000, 40000, 50000])

# Model
model = DecisionTreeRegressor()

# Train
model.fit(X, y)

# Prediction
prediction = model.predict([[3]])

print("Predicted Salary:", prediction)
```

---

# Libraries Used

* Python
* NumPy
* Scikit-learn
* Jupyter Notebook

Install dependencies:

```
pip install numpy scikit-learn
```

---

# How to Run the Project

1. Clone the repository

```
git clone https://github.com/your-username/your-repository-name.git
```

2. Open the project folder

```
cd your-repository-name
```

3. Launch Jupyter Notebook

```
jupyter notebook
```

4. Run the notebooks

* Decision_tree_classifier.ipynb
* Decision_tree_regressor.ipynb

---

# Concepts Covered

* Decision Tree Algorithm
* Supervised Machine Learning
* Classification
* Regression
* Model Training
* Model Prediction

---

# Random Forest Machine Learning Models

This project demonstrates the implementation of **Random Forest algorithms** using Python and Scikit-learn.  
It includes two machine learning tasks:

1. **Random Forest Classifier**
2. **Random Forest Regressor**

Both models are implemented using simple datasets to demonstrate the basic workflow of machine learning.

---

# Project Structure

```
.
├── Random_forest_classifier.ipynb
├── Random_forest_regressor.ipynb
└── README.md
```

---

# 1. Random Forest Classifier

## Overview
The **Random Forest Classifier** is a supervised machine learning algorithm used for **classification problems**.  
It builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.

## Example Problem
Predict whether a student will **Pass or Fail** based on input features.

## Dataset Example

| Input Feature | Output |
|---------------|-------|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Pass |
| 5 | Pass |

Where:
- `0 = Fail`
- `1 = Pass`

## Steps Implemented

1. Import required libraries
2. Create sample dataset
3. Initialize Random Forest Classifier
4. Train the model
5. Make predictions

## Libraries Used

```
scikit-learn
numpy
```

## Example Code

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

prediction = model.predict([[3.5]])
print(prediction)
```

---

# 2. Random Forest Regressor

## Overview
The **Random Forest Regressor** is used for **regression problems**, where the output is a continuous value.

It combines predictions from multiple decision trees to produce a more accurate result.

## Example Problem
Predict **salary** based on years of experience.

## Dataset Example

| Experience | Salary |
|-----------|--------|
| 1 | 10000 |
| 2 | 20000 |
| 3 | 30000 |
| 4 | 40000 |
| 5 | 50000 |

## Steps Implemented

1. Import required libraries
2. Create sample dataset
3. Initialize Random Forest Regressor
4. Train the model
5. Predict salary for new input

## Libraries Used

```
scikit-learn
numpy
```

## Example Code

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10000, 20000, 30000, 40000, 50000])

model = RandomForestRegressor(n_estimators=10)
model.fit(X, y)

prediction = model.predict([[3.5]])
print(prediction)
```

---

# Installation

Install required libraries before running the notebooks.

```
pip install numpy scikit-learn
```

---

# How to Run

1. Clone the repository

```
git clone <repository-link>
```

2. Navigate to the project folder

```
cd project-folder
```

3. Open the notebooks in **Jupyter Notebook** or **VS Code**

4. Run the cells step by step.

---

# Technologies Used

- Python
- Scikit-learn
- NumPy
- Jupyter Notebook

---

# Learning Outcomes

From this project you will learn:

- Basics of **Random Forest Algorithm**
- Difference between **Classification and Regression**
- Model training and prediction
- Using **Scikit-learn** for machine learning

---

# Author

Vishal Kumar  
Artificial Intelligence and Data Science
---
