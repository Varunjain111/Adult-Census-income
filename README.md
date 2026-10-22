# Adult-Census-income
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

display(Markdown("1. Data Loading & Preprocessing"))
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income']
df = pd.read_csv(url, names=cols, skipinitialspace=True)

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)

X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
print("Data successfully loaded and preprocessed.")

display(Markdown("---"))
display(Markdown("Part 1: OLS vs Logistic Regression (Build & Compare)"))

ols = LinearRegression()
ols.fit(X_train_processed, y_train)
ols_preds = ols.predict(X_test_processed)
ols_median = np.median(ols_preds)
ols_class = (ols_preds >= ols_median).astype(int)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_processed, y_train)
lr_probs = lr.predict_proba(X_test_processed)[:, 1]
lr_class = (lr_probs >= 0.5).astype(int)

agreement = (ols_class == lr_class)
mismatch_count = np.sum(~agreement)
mismatch_pct = mismatch_count / len(y_test) * 100

part1_df = pd.DataFrame({
    "Metric": ["Linear Regression Median Threshold", "Mismatched Cases", "Mismatch Percentage"],
    "Value": [round(ols_median, 4), mismatch_count, f"{mismatch_pct:.2f}%"]
})
display(part1_df)

cm = confusion_matrix(ols_class, lr_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['LR Low Risk (0)', 'LR High Risk (1)'], yticklabels=['OLS Low Risk (0)', 'OLS High Risk (1)'])
plt.title('Confusion Matrix: OLS vs Logistic Regression')
plt.ylabel('OLS Predictions')
plt.xlabel('Logistic Regression Predictions')
plt.show()

display(Markdown("---"))
display(Markdown("Part 2.1: Simpson's Paradox Hunt (Subgroup Analysis)"))

sex_male_idx = feature_names.index('sex_Male')
age_idx = feature_names.index('age')

male_train_idx = X_train_processed[:, sex_male_idx] == 1
female_train_idx = X_train_processed[:, sex_male_idx] == 0

lr_full = LogisticRegression(max_iter=1000).fit(X_train_processed, y_train)
lr_male = LogisticRegression(max_iter=1000).fit(X_train_processed[male_train_idx], y_train.iloc[male_train_idx])
lr_female = LogisticRegression(max_iter=1000).fit(X_train_processed[female_train_idx], y_train.iloc[female_train_idx])

subgroup_data = {
    "Model": ["Full Model", "Male Subgroup", "Female Subgroup"],
    "Age Coefficient": [lr_full.coef_[0][age_idx], lr_male.coef_[0][age_idx], lr_female.coef_[0][age_idx]]
}
subgroup_df = pd.DataFrame(subgroup_data)
display(subgroup_df)

plt.figure(figsize=(7, 4))
sns.barplot(x='Model', y='Age Coefficient', data=subgroup_df, palette='viridis')
plt.title('Effect of Age on Income Across Gender Subgroups')
plt.ylabel('Logistic Regression Coefficient')
plt.show()

display(Markdown("---"))
display(Markdown("Part 2.2: Omitted-Variable Bias"))

ed_idx = feature_names.index('education_num')
X_train_no_ed = np.delete(X_train_processed, ed_idx, axis=1)
lr_no_ed = LogisticRegression(max_iter=1000).fit(X_train_no_ed, y_train)

shift_vars = ['age', 'hours_per_week', 'capital_gain', 'sex_Male']
ovb_results = []

for var in shift_vars:
    i = feature_names.index(var)
    idx_no_ed = i if i < ed_idx else i - 1
    c_full = lr_full.coef_[0][i]
    c_no_ed = lr_no_ed.coef_[0][idx_no_ed]
    pct_change = abs((c_no_ed - c_full) / c_full) * 100
    ovb_results.append([var, c_full, c_no_ed, pct_change])

ovb_df = pd.DataFrame(ovb_results, columns=['Predictor', 'Full Model Coef', 'Reduced Model Coef', 'Percentage Change (%)'])
display(ovb_df)

plt.figure(figsize=(8, 4))
sns.barplot(x='Percentage Change (%)', y='Predictor', data=ovb_df, palette='magma')
plt.title('Percentage Shift in Coefficients after Dropping Education')
plt.xlabel('% Change')
plt.show()

display(Markdown("---"))
display(Markdown("Part 2.4: Fairness Trade-Off Analysis"))

male_test_idx = X_test['sex'].str.strip() == 'Male'
female_test_idx = X_test['sex'].str.strip() == 'Female'
fairness_metrics = []

for t in np.arange(0.05, 1.0, 0.1):
    preds = (lr_probs >= t).astype(int)

    p_male = np.mean(preds[male_test_idx])
    p_female = np.mean(preds[female_test_idx])
    dp = p_female / (p_male + 1e-9)

    tpr_m = np.sum((preds[male_test_idx] == 1) & (y_test[male_test_idx] == 1)) / (np.sum(y_test[male_test_idx] == 1) + 1e-9)
    tpr_f = np.sum((preds[female_test_idx] == 1) & (y_test[female_test_idx] == 1)) / (np.sum(y_test[female_test_idx] == 1) + 1e-9)
    fpr_m = np.sum((preds[male_test_idx] == 1) & (y_test[male_test_idx] == 0)) / (np.sum(y_test[male_test_idx] == 0) + 1e-9)
    fpr_f = np.sum((preds[female_test_idx] == 1) & (y_test[female_test_idx] == 0)) / (np.sum(y_test[female_test_idx] == 0) + 1e-9)
    eo = min(tpr_f / (tpr_m + 1e-9), fpr_f / (fpr_m + 1e-9))

    pp_m = np.sum((preds[male_test_idx] == 1) & (y_test[male_test_idx] == 1)) / (np.sum(preds[male_test_idx] == 1) + 1e-9)
    pp_f = np.sum((preds[female_test_idx] == 1) & (y_test[female_test_idx] == 1)) / (np.sum(preds[female_test_idx] == 1) + 1e-9)
    pp = pp_f / (pp_m + 1e-9)

    fairness_metrics.append([round(t, 2), dp, eo, pp])

fairness_df = pd.DataFrame(fairness_metrics, columns=['Threshold', 'Demographic Parity (DP)', 'Equalized Odds (EO)', 'Predictive Parity (PP)'])
display(fairness_df.round(4))

plt.figure(figsize=(9, 5))
plt.plot(fairness_df['Threshold'], fairness_df['Demographic Parity (DP)'], marker='o', label='Demographic Parity')
plt.plot(fairness_df['Threshold'], fairness_df['Equalized Odds (EO)'], marker='s', label='Equalized Odds')
plt.plot(fairness_df['Threshold'], fairness_df['Predictive Parity (PP)'], marker='^', label='Predictive Parity')
plt.axhline(1.0, color='red', linestyle='--', alpha=0.6, label='Ideal Parity (1.0)')
plt.title('Fairness Trade-offs Across Decision Thresholds')
plt.xlabel('Decision Threshold')
plt.ylabel('Metric Ratio (Female/Male)')
plt.legend()
plt.grid(True)
plt.show()
