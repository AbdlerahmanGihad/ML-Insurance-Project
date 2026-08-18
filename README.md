# 🏥 Medical Insurance Cost Prediction

A complete Machine Learning project for predicting **medical insurance charges** using multiple regression algorithms, hyperparameter tuning, and a Streamlit web application.

---

## 📌 Project Overview

The goal of this project is to build and compare different Machine Learning regression models to predict the medical insurance cost for an individual based on personal and demographic information.

The project follows a complete Machine Learning workflow:

**Data Collection → Data Preprocessing → Exploratory Data Analysis → Model Training → Hyperparameter Tuning → Model Evaluation → Model Saving → Deployment**

---

## 🎯 Objectives

* Analyze the Medical Insurance dataset.
* Perform data preprocessing and preparation.
* Train multiple regression models.
* Compare the performance of different models.
* Optimize models using `GridSearchCV`.
* Evaluate models using multiple regression metrics.
* Save trained models using Joblib.
* Deploy the prediction system using Streamlit.

---

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**.

### Features

| Feature    | Description            |
| ---------- | ---------------------- |
| `age`      | Age of the individual  |
| `sex`      | Gender                 |
| `bmi`      | Body Mass Index        |
| `children` | Number of children     |
| `smoker`   | Smoking status         |
| `region`   | Residential region     |
| `charges`  | Medical insurance cost |

### Target Variable

```text
charges
```

The target variable represents the medical insurance cost that the model attempts to predict.

---

## 🤖 Machine Learning Models

Three regression algorithms were implemented and compared:

### 1. Linear Regression

A simple regression algorithm used as a baseline model.

```python
LinearRegression()
```

### 2. Decision Tree Regressor

A tree-based regression algorithm capable of learning nonlinear relationships.

Hyperparameters were optimized using `GridSearchCV`.

### 3. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

Hyperparameters were also optimized using `GridSearchCV`.

---

## 🔍 Hyperparameter Tuning

`GridSearchCV` was used to find better hyperparameters for the Decision Tree and Random Forest models.

This helps improve model performance while reducing the risk of selecting unsuitable parameters manually.

Example parameters include:

```text
max_depth
min_samples_split
n_estimators
```

---

## 📈 Model Evaluation

The models were evaluated using the following metrics:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error — MSE

Penalizes larger prediction errors more heavily.

### Root Mean Squared Error — RMSE

The square root of MSE and provides the error in the same unit as the target variable.

### R² Score

Measures how well the model explains the variation in the target variable.

A higher R² score generally indicates better performance.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## 📁 Project Structure

```text
ML-Insurance-Project/
│
├── data/
│   └── raw/
│       └── insurance.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── notebooks/
│   └── insurance_ml.ipynb
│
├── src/
│   └── ...
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> File and folder names can be adjusted to match the actual project structure.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AbdlerahmanGihad/ML-Insurance-Project.git
```

### 2. Navigate to the Project

```bash
cd ML-Insurance-Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run the Streamlit Application

```bash
streamlit run app.py
```

After running the command, Streamlit will open the application in your browser.

The user can enter the required information and receive a predicted medical insurance charge.

---

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

The deployed application provides an interactive interface where users can enter their information and obtain an insurance cost prediction.

---

## 🧠 Machine Learning Workflow

```text
                Dataset
                   │
                   ▼
          Data Preprocessing
                   │
                   ▼
        Exploratory Data Analysis
                   │
                   ▼
            Train / Test Split
                   │
                   ▼
       ┌───────────┼───────────┐
       ▼           ▼           ▼
 Linear        Decision      Random
Regression       Tree         Forest
       │           │           │
       └───────────┼───────────┘
                   ▼
            Model Evaluation
                   │
                   ▼
            GridSearchCV
                   │
                   ▼
            Best Model(s)
                   │
                   ▼
             Joblib Models
                   │
                   ▼
             Streamlit App
```

---

## 📌 Results

The performance of the implemented models is compared using:

| Model             | MAE | MSE | RMSE | R² |
| ----------------- | --: | --: | ---: | -: |
| Linear Regression |   — |   — |    — |  — |
| Decision Tree     |   — |   — |    — |  — |
| Random Forest     |   — |   — |    — |  — |

> Replace the `—` values with the actual results from your project.

---

## 💡 Key Learning Outcomes

Through this project, we learned how to:

* Work with a real-world dataset.
* Perform data preprocessing.
* Build regression models.
* Compare different Machine Learning algorithms.
* Use evaluation metrics.
* Perform hyperparameter tuning with `GridSearchCV`.
* Save trained models using Joblib.
* Build a Machine Learning web application using Streamlit.
* Prepare a Machine Learning project for deployment.

---

## 🚀 Future Improvements

Possible improvements include:

* Adding XGBoost, CatBoost, or LightGBM.
* Improving feature engineering.
* Adding more visualizations.
* Performing advanced hyperparameter optimization.
* Improving the Streamlit UI.
* Adding model explainability using SHAP.
* Deploying the application online.

---

## 👨‍💻 Author

### Abdelrahman Gihad

Artificial Intelligence Student

GitHub:
https://github.com/AbdlerahmanGihad

---

## ⭐ Project

If you find this project useful, consider giving it a ⭐ on GitHub.

**Made with Python & Machine Learning 🚀**
