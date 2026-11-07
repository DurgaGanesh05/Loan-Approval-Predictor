# 🏦 Loan Approval Predictor

A machine learning project that predicts loan approval (`LoanApproved`) from applicant profile data.

This repository currently contains a single end-to-end training script (`main2.py`) that:
- loads and cleans an impure dataset,
- performs exploratory analysis and visualizations,
- trains two classification models,
- evaluates both models, and
- prints final model comparison metrics.

---

## 📁 Project Structure

```
Loan-Approval-Predictor/
├── loan_approval_ - loan_approval_impure.csv.csv   # Input dataset (contains missing/corrupted values)
├── main2.py                                        # Data cleaning, EDA, training, and evaluation script
├── README.md                                       # Project documentation
└── DSML.pdf                                        # Certificate/document
```

---

## ⚙️ Tech Stack

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔍 What the Script Does

### 1) Data Loading
- Reads `loan_approval_ - loan_approval_impure.csv.csv`.

### 2) Data Cleaning
- Drops rows where `LoanApproved` is missing.
- Removes rows with invalid target labels (keeps only `0` and `1`).
- Fills missing values in:
  - `ApplicantIncome`
  - `LoanAmount`
  - `CreditScore`
  using median imputation.

### 3) EDA and Visualization
- Prints summary statistics.
- Generates histograms for numeric columns.
- Shows approval rates by:
  - `Education`
  - `SelfEmployed`
- Displays a correlation heatmap.

### 4) Feature Preparation
- Label-encodes categorical fields:
  - `Education`
  - `SelfEmployed`
- Uses these input features:
  - `ApplicantIncome`, `LoanAmount`, `CreditScore`, `Education`, `SelfEmployed`

### 5) Model Training and Evaluation
- Splits data with `test_size=0.2` and `random_state=42`.
- Trains:
  - Logistic Regression
  - Decision Tree Classifier
- Prints:
  - classification report,
  - confusion matrix,
  - final accuracy comparison.

---

## 📊 Latest Model Performance

From the current code and dataset (same split and seed as script):

| Model | Accuracy |
|---|---|
| Logistic Regression | **0.50** |
| Decision Tree | **0.53** |

✅ Current best model in this project version: **Decision Tree**.

---

## 🚀 Run the Project

### 1) Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 2) Execute the script

```bash
python main2.py
```

> If you run in a headless environment, use:
>
> ```bash
> MPLBACKEND=Agg python main2.py
> ```

---

## 📝 Notes

- The dataset intentionally includes noise/missing/corrupted records, and the script handles them before training.
- Model accuracy is expected to change if dataset content, preprocessing, split ratio, or random seed changes.

---

## 👤 Author

**Seeram Venkata Durga Ganesh**  
🔗 [GitHub](https://github.com/DurgaGanesh05)
