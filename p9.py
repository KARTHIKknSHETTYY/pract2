import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load and clean
df = fetch_openml('adult', version=2, as_frame=True).frame
df = df.replace('?', pd.NA).dropna()
df['target'] = (df['class'] == '>50K').astype(int)
df['sex'] = (df['sex'] == 'Male').astype(int)

# Features and split
X = df[['age', 'sex']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train and check bias
model = LogisticRegression().fit(X_train, y_train)
preds = model.predict(X_test)
print("Male rate:", preds[X_test['sex']==1].mean(), "| Female rate:", preds[X_test['sex']==0].mean())

# Mitigate: balance data by upsampling
train = X_train.copy()
train['target'] = y_train
males, females = train[train['sex']==1], train[train['sex']==0]
minority = females if len(females) < len(males) else males
upsampled = resample(minority, replace=True, n_samples=max(len(males), len(females)), random_state=42)
balanced = pd.concat([males, females, upsampled]).drop_duplicates()

# Retrain and check again
model2 = LogisticRegression().fit(balanced[['age', 'sex']], balanced['target'])
preds2 = model2.predict(X_test)
print("Post-balance Male:", preds2[X_test['sex']==1].mean(), "| Female:", preds2[X_test['sex']==0].mean())
