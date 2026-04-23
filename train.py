import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from fairlearn.metrics import demographic_parity_difference, MetricFrame
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

print("1. Loading Dataset...")
# Dataset load kar rahe hain (Make sure train.csv is in the same folder)
df = pd.read_csv('train.csv')

# Hum sirf wo clinical features le rahe hain jo website pe input form me honge
selected_features = [
    'age_at_hct', 'karnofsky_score', 'comorbidity_score', 'donor_age',
    'dri_score', 'conditioning_intensity', 'race_group', 'sex_match', 
    'graft_type', 'donor_related'
]

# Target variable (EFS = Event Free Survival)
# Let's assume efs = 1 means Survived/Event-free, efs = 0 means Event occurred
df = df.dropna(subset=['efs']) # Drop rows where target is missing
X = df[selected_features]
y = df['efs']

print("2. Splitting Data into Train and Test...")
# 80% data training ke liye, 20% testing ke liye
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Building Preprocessing Pipeline...")
# Numeric aur Categorical columns ko alag alag process karna
numeric_features = ['age_at_hct', 'karnofsky_score', 'comorbidity_score', 'donor_age']
categorical_features = ['dri_score', 'conditioning_intensity', 'race_group', 'sex_match', 'graft_type', 'donor_related']

# Numeric data ki missing values ko median se fill karna aur scale karna
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical data ki missing values ko mode se fill karna aur One-Hot Encode karna
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("4. Training LightGBM Model (with Fairness logic)...")
# Pipeline mein preprocessor aur LightGBM model dono ko jor diya hai
# 'class_weight='balanced' use kar rahe hain taake dataset imbalance theek ho
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=7,
        class_weight='balanced',
        random_state=42
    ))
])

# Model ko train kar rahe hain
model_pipeline.fit(X_train, y_train)

print("5. Evaluating Model Accuracy...")
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"--- Performance Metrics ---")
print(f"AUC-ROC Score: {auc:.4f}")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")

print("\n6. Evaluating Fairness (Demographic Parity)...")
# Hum race_group ki base par bias check kar rahe hain
sensitive_features = X_test['race_group'].fillna('Unknown')

# Demographic Parity Difference (Threshold <= 0.10 hona chahiye)
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
print(f"Demographic Parity Difference (Race): {dpd:.4f}")

if dpd <= 0.10:
    print("STATUS: PASS (Model is Fair!)")
else:
    print("STATUS: WARNING (Fairness mitigation needs tuning)")

print("\n7. Saving Model for FastAPI and Streamlit...")
joblib.dump(model_pipeline, 'hct_survival_model.pkl')
print("Model saved successfully as 'hct_survival_model.pkl'!")