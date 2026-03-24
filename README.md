# 🏦 Loan Approval Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A machine learning web application that predicts whether a loan application will be **approved or rejected** based on applicant details. This project demonstrates an end-to-end ML pipeline — from data preprocessing and model training to deploying a live prediction service via a Flask web application.

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎬 Demo

> Enter applicant information (income, credit history, loan amount, etc.) and get an instant loan approval prediction.

```
Input:  Gender=Male, Married=Yes, Dependents=1, Education=Graduate,
        Self_Employed=No, ApplicantIncome=5000, CoapplicantIncome=2000,
        LoanAmount=150, Loan_Amount_Term=360, Credit_History=1,
        Property_Area=Urban

Output: ✅ Loan Approved
```

---

## ✨ Features

- 📊 **Exploratory Data Analysis (EDA)** — visualizes distributions, correlations, and missing value patterns
- 🧹 **Automated Data Preprocessing** — handles missing values, encodes categorical variables, and scales numerical features
- 🤖 **Multiple ML Models** — trains and compares Logistic Regression, Decision Tree, Random Forest, and SVM
- 📈 **Model Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC curves
- 🌐 **Flask Web Application** — clean, user-friendly UI to submit loan applications and view predictions
- 💾 **Model Persistence** — trained model saved with `joblib` / `pickle` for instant inference

---

## 📦 Dataset

The dataset used is the **Loan Prediction Dataset** (commonly sourced from [Kaggle](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)).

### Features

| Feature | Type | Description |
|---|---|---|
| `Loan_ID` | Object | Unique loan reference ID |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Applicant's marital status (Yes / No) |
| `Dependents` | Categorical | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Whether self-employed (Yes / No) |
| `ApplicantIncome` | Numerical | Applicant's monthly income |
| `CoapplicantIncome` | Numerical | Co-applicant's monthly income |
| `LoanAmount` | Numerical | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Numerical | Term of the loan (in months) |
| `Credit_History` | Binary | Credit history meets guidelines (1 / 0) |
| `Property_Area` | Categorical | Urban / Semi-Urban / Rural |
| `Loan_Status` *(target)* | Binary | Loan approved (Y) or rejected (N) |

### Dataset Statistics

- **Training samples:** ~614 records
- **Target classes:** `Y` (Approved) · `N` (Rejected)
- **Class distribution:** ~68% Approved, ~32% Rejected

---

## 📁 Project Structure

```
Loan-Approval-Predictor/
├── dataset/
│   ├── train.csv                  # Training dataset
│   └── test.csv                   # Test dataset
│
├── notebooks/
│   └── loan_approval_analysis.ipynb  # EDA and model training notebook
│
├── models/
│   └── loan_model.pkl             # Saved trained model
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom styles
│   └── images/
│       └── banner.png             # App banner
│
├── templates/
│   ├── index.html                 # Home / input form
│   └── result.html                # Prediction result page
│
├── app.py                         # Flask application entry point
├── model.py                       # Model training and evaluation script
├── preprocess.py                  # Data preprocessing utilities
├── requirements.txt               # Python dependencies
├── DSML.pdf                       # Training completion certificate
└── README.md                      # Project documentation
```

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Web Framework** | Flask |
| **Model Persistence** | Joblib / Pickle |
| **Frontend** | HTML5, CSS3, Bootstrap 5 |

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- Drop `Loan_ID` (non-informative identifier)
- Impute missing values:
  - **Categorical** columns → mode (most frequent value)
  - **Numerical** columns → median
- Encode categorical variables using `LabelEncoder`
- Scale numerical features using `StandardScaler`

### 2. Exploratory Data Analysis
- Distribution plots for income, loan amount, and loan term
- Count plots for categorical features vs. loan status
- Correlation heatmap of all features
- Handling class imbalance insights

### 3. Model Training
Four classifiers are trained and compared:

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier |
| **Decision Tree** | Interpretable tree-based model |
| **Random Forest** | Ensemble of decision trees |
| **Support Vector Machine** | Kernel-based classification |

### 4. Model Evaluation
Models are evaluated on:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1)
- **ROC-AUC Score**

The best-performing model is saved for inference.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/DurgaGanesh05/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model.py
```

This will:
1. Load and preprocess the dataset
2. Train all models and display their accuracy
3. Save the best model as `models/loan_model.pkl`

---

## 🚀 Usage

### Run the Flask Web App

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000/
```

### Fill the Application Form

Enter the following details in the form:

- **Gender** — Male / Female
- **Marital Status** — Yes / No
- **Number of Dependents** — 0 / 1 / 2 / 3+
- **Education** — Graduate / Not Graduate
- **Self Employed** — Yes / No
- **Applicant Income** — Monthly income (in ₹ or $)
- **Coapplicant Income** — Co-applicant's monthly income
- **Loan Amount** — Requested loan amount (in thousands)
- **Loan Term** — Repayment period in months (e.g., 360)
- **Credit History** — 1 (good) / 0 (bad or none)
- **Property Area** — Urban / Semi-Urban / Rural

Click **Predict** to view the result.

---

## 📊 Model Performance

> Results on the validation/test split (80/20):

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~80% | ~0.82 | ~0.93 | ~0.87 | ~0.79 |
| Decision Tree | ~76% | ~0.79 | ~0.88 | ~0.83 | ~0.70 |
| **Random Forest** | **~82%** | **~0.84** | **~0.94** | **~0.89** | **~0.81** |
| SVM | ~79% | ~0.81 | ~0.92 | ~0.86 | ~0.77 |

> ✅ **Random Forest** achieves the best overall performance and is used as the default model in the web application.

---

## 📡 API Reference

The Flask app also exposes a simple JSON prediction endpoint:

### `POST /predict`

**Request Body (JSON):**

```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 1,
  "Education": 0,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 2
}
```

**Response (JSON):**

```json
{
  "prediction": "Approved",
  "confidence": 0.87
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Seeram Venkata Durga Ganesh**  
🎓 Lovely Professional University  
📜 Certified in Machine Learning with Data Science — CipherSchools (Jun–Jul 2025)  
🔗 [GitHub](https://github.com/DurgaGanesh05)

---

> *"Automating smarter lending decisions with the power of Machine Learning."*
