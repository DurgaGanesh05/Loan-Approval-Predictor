
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================================
# 1. LOAD DATASET
# ==============================================

df = pd.read_csv('loan_approval_ - loan_approval_impure.csv.csv')

print(df.head(10))
print("Data loaded successfully.")


# ==============================================
# 2. DATA CLEANING
# ==============================================

# Remove rows where the target value is missing
df_cleaned = df.dropna(subset=['LoanApproved']).copy()

# Keep only valid target values: 0 and 1
# Other values are treated as corrupted labels
invalid_rows = df_cleaned[~df_cleaned['LoanApproved'].isin([0, 1])]

if not invalid_rows.empty:
    print("\nInvalid target records:")
    print(invalid_rows)

    df_cleaned = df_cleaned[df_cleaned['LoanApproved'].isin([0, 1])].copy()

# Fill missing numerical values using median
columns = ['ApplicantIncome', 'LoanAmount', 'CreditScore']

for col in columns:
    median_value = df_cleaned[col].median()
    df_cleaned[col] = df_cleaned[col].fillna(median_value)

# Check missing values
print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum())

assert df_cleaned[columns].isnull().sum().sum() == 0


# ==============================================
# 3. DATA EXPLORATION & VISUALIZATION
# ==============================================

# Display numerical statistics
print("\nNumerical Data Summary:")
print(df_cleaned[columns].describe())

# Display distributions using histograms
sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(columns):
    sns.histplot(data=df_cleaned, x=col, kde=True, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# Approval rate based on Education
print("\nApproval Rate by Education:")
print(df_cleaned.groupby('Education')['LoanApproved'].mean())

# Approval rate based on Self Employment
print("\nApproval Rate by Self Employment:")
print(df_cleaned.groupby('SelfEmployed')['LoanApproved'].mean())

# Correlation heatmap
sns.heatmap(
    df_cleaned.select_dtypes(include='number').corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap')
plt.show()


# ==============================================
# 4. ENCODING CATEGORICAL DATA
# ==============================================

df_encoded = df_cleaned.copy()

encoder = LabelEncoder()

df_encoded['Education'] = encoder.fit_transform(df_encoded['Education'])
df_encoded['SelfEmployed'] = encoder.fit_transform(df_encoded['SelfEmployed'])


# ==============================================
# 5. SELECT FEATURES AND TARGET
# ==============================================

# Input features
X = df_encoded[
    ['ApplicantIncome', 'LoanAmount', 'CreditScore',
     'Education', 'SelfEmployed']
]

# Output / target
y = df_encoded['LoanApproved'].astype(int)

# Check target classes
print("\nTarget Class Distribution:")
print(y.value_counts())


# ==============================================
# 6. SPLIT DATASET
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Classes:", sorted(y_train.unique()))
print("Testing Classes:", sorted(y_test.unique()))

assert set(y_train.unique()) == {0, 1}
assert set(y_test.unique()) == {0, 1}


# ==============================================
# 7. MODEL TRAINING
# ==============================================

# Create models
logreg = LogisticRegression(max_iter=1000)
dtree = DecisionTreeClassifier(random_state=42)

# Train both models
logreg.fit(X_train, y_train)
dtree.fit(X_train, y_train)


# ==============================================
# 8. MODEL PREDICTION
# ==============================================

y_pred_log = logreg.predict(X_test)
y_pred_tree = dtree.predict(X_test)


# ==============================================
# 9. MODEL EVALUATION
# ==============================================

print("\nLOGISTIC REGRESSION RESULTS")

print(classification_report(y_test, y_pred_log, labels=[0, 1]))

cm_log = confusion_matrix(y_test, y_pred_log, labels=[0, 1])
print("Confusion Matrix:")
print(cm_log)

assert cm_log.shape == (2, 2)


print("\nDECISION TREE RESULTS")

print(classification_report(y_test, y_pred_tree, labels=[0, 1]))

cm_tree = confusion_matrix(y_test, y_pred_tree, labels=[0, 1])
print("Confusion Matrix:")
print(cm_tree)

assert cm_tree.shape == (2, 2)


# ==============================================
# 10. FINAL MODEL COMPARISON
# ==============================================

acc_log = accuracy_score(y_test, y_pred_log)
acc_tree = accuracy_score(y_test, y_pred_tree)

print("\nFINAL MODEL COMPARISON")

print(f"Logistic Regression Accuracy: {acc_log:.2f}")
print(f"Decision Tree Accuracy: {acc_tree:.2f}")

if acc_log > acc_tree:
    print("Logistic Regression performed better.")
else:
    print("Decision Tree performed better.")
