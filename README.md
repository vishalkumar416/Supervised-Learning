# Supervised Machine Learning Algorithms (Scikit-Learn)

This repository demonstrates the implementation of several **Supervised Machine Learning algorithms** using **Python and Scikit-learn**.
The project focuses on both **classification and regression models** and provides simple examples to understand how different machine learning algorithms work.

Each algorithm is implemented using **Jupyter Notebooks**, allowing users to explore the workflow of building, training, and evaluating machine learning models.

---

# 📌 Algorithms Included

The repository contains implementations of the following machine learning algorithms:

### Classification Algorithms

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Classifier (SVC)
* Gradient Boosting Classifier
* K-Nearest Neighbors (KNN) Classifier

### Regression Algorithms

* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)
* Gradient Boosting Regressor
* K-Nearest Neighbors (KNN) Regressor

---

# 📂 Project Structure

```
Machine-Learning-Algorithms/
│
├── Logistic_regression.ipynb
├── Decision_tree_classifier.ipynb
├── Decision_tree_regressor.ipynb
├── Random_forest_classifier.ipynb
├── Random_forest_regressor.ipynb
├── SVC.ipynb
├── SVR.ipynb
├── GradientBoostingClassifier.ipynb
├── GradientBoostingRegressor.ipynb
├── KNeighborsClassifier.ipynb
├── KNeighborsRegressor.ipynb
│
└── README.md
```

---

# 🧠 Algorithm Overview

## Logistic Regression

Logistic Regression is a **supervised machine learning algorithm** used for **binary classification problems**.
It predicts the probability that an input belongs to a particular class using the **sigmoid function**.

### Logistic Function

P(Y=1) = 1 / (1 + e^(-z))

Where:

z = b0 + b1X

### Example Dataset

| Hours Studied | Result |
| ------------- | ------ |
| 1             | Fail   |
| 2             | Fail   |
| 3             | Fail   |
| 4             | Pass   |
| 5             | Pass   |

Where:

* 0 = Fail
* 1 = Pass

### Example Code

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

# Model
model = LogisticRegression()

# Train
model.fit(X, y)

# Prediction
prediction = model.predict([[3.5]])
probability = model.predict_proba([[3.5]])

print("Prediction:", prediction)
print("Probability:", probability)
```

---

## Decision Tree

A **Decision Tree** is a supervised learning algorithm used for **classification and regression tasks**.
It splits the dataset into smaller subsets based on feature values, forming a tree-like structure of decision rules.

### Key Advantages

* Easy to understand and interpret
* Works with both numerical and categorical data
* Requires minimal preprocessing

---

### Decision Tree Classifier Example

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([[3.5]])

print("Prediction:", prediction)
```

---

### Decision Tree Regressor Example

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10000, 20000, 30000, 40000, 50000])

model = DecisionTreeRegressor()
model.fit(X, y)

prediction = model.predict([[3]])

print("Predicted Salary:", prediction)
```

---

## Random Forest

Random Forest is an **ensemble learning algorithm** that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

### Random Forest Classifier Example

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

### Random Forest Regressor Example

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

## Support Vector Machine (SVM)

Support Vector Machines are powerful supervised learning algorithms used for **classification and regression tasks**.
They work by identifying the **optimal hyperplane** that separates data points with the maximum margin.

---

### Support Vector Classifier (SVC)

```python
from sklearn.svm import SVC
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

model = SVC(kernel='linear')
model.fit(X, y)

prediction = model.predict([[3.5]])
print(prediction)
```

---

### Support Vector Regressor (SVR)

```python
from sklearn.svm import SVR
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10000, 20000, 30000, 40000, 50000])

model = SVR(kernel='rbf')
model.fit(X, y)

prediction = model.predict([[3.5]])
print(prediction)
```

---

## Gradient Boosting

Gradient Boosting is an **ensemble learning technique** where models are built sequentially.
Each new model attempts to correct the errors of the previous model.

---

### Gradient Boosting Classifier Example

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### Gradient Boosting Regressor Example

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=200, n_features=4, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## K-Nearest Neighbors (KNN)

KNN is a simple supervised learning algorithm that predicts outcomes based on the **nearest data points in the dataset**.

---

### KNN Classifier Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### KNN Regressor Example

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=200, n_features=4, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# ⚙️ Installation

Install the required libraries before running the notebooks.

```
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install jupyter
```

---

# 🚀 How to Run the Project

1. Clone the repository

```
git clone https://github.com/your-username/machine-learning-algorithms.git
```

2. Navigate to the project directory

```
cd machine-learning-algorithms
```

3. Launch Jupyter Notebook

```
jupyter notebook
```

4. Open any notebook and run the cells step by step.

---

# 🎯 Learning Outcomes

By working through this project, you will gain an understanding of:

* Fundamental concepts of **supervised machine learning**
* Differences between **classification and regression algorithms**
* Implementing models using **Scikit-learn**
* Training, testing, and evaluating machine learning models
* Practical applications of common machine learning algorithms

---

# 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# 👨‍💻 Author

**Vishal Kumar**

B-Tech in Artificial Intelligence and Data Science
