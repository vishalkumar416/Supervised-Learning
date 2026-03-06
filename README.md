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

## 👨‍💻 Author
**Vishal Kumar**

B.Tech (AI & Data Science)  
---

⭐ If you found this project helpful, consider giving it a star on GitHub.
