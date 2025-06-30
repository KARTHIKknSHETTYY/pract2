import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load UCI Adult dataset
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame

# Basic cleaning: drop rows with missing values
df = df.replace('?', pd.NA).dropna()

# Binary target: income >50K or <=50K
df['target'] = df['class'].apply(lambda x: 1 if x == '>50K' else 0)

# Encode 'sex' as binary: Male = 1, Female = 0
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)

# Simple feature set: use 'age' and 'sex'
X = df[['age', 'sex']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train base model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Base accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Check for bias: difference in positive prediction rate for males vs. females
X_test['sex'] = X_test['sex'].astype(int)
male_pred_rate = y_pred[X_test['sex'] == 1].mean()
female_pred_rate = y_pred[X_test['sex'] == 0].mean()
print(f"Positive prediction rate (Male): {male_pred_rate:.2f}")
print(f"Positive prediction rate (Female): {female_pred_rate:.2f}")

# Mitigate bias by balancing males and females in training data
df_train = X_train.copy()
df_train['target'] = y_train

df_male = df_train[df_train['sex'] == 1]
df_female = df_train[df_train['sex'] == 0]

# Upsample minority group (female)
if len(df_male) > len(df_female):
    df_female_upsampled = resample(df_female,
                                   replace=True,
                                   n_samples=len(df_male),
                                   random_state=42)
    df_balanced = pd.concat([df_male, df_female_upsampled])
else:
    df_male_upsampled = resample(df_male,
                                 replace=True,
                                 n_samples=len(df_female),
                                 random_state=42)
    df_balanced = pd.concat([df_male_upsampled, df_female])

# Retrain on balanced data
X_train_bal = df_balanced[['age', 'sex']]
y_train_bal = df_balanced['target']

model_bal = LogisticRegression()
model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = model_bal.predict(X_test)

print(f"Balanced accuracy: {accuracy_score(y_test, y_pred_bal):.2f}")

# Check bias again
male_pred_rate_bal = y_pred_bal[X_test['sex'] == 1].mean()
female_pred_rate_bal = y_pred_bal[X_test['sex'] == 0].mean()
print(f"Positive prediction rate after balancing (Male): {male_pred_rate_bal:.2f}")
print(f"Positive prediction rate after balancing (Female): {female_pred_rate_bal:.2f}")
