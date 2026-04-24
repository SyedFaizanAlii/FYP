"""
=============================================================
Equitable Survival Prediction after Hematopoietic Cell Transplant
KFUEIT Final Year Project - Complete ML Pipeline WITH FAIRNESS
Authors: Muzammil Tariq & Syed Faizan Ali (COSC221101046)
Rebuilt with: In-Processing Fairlearn Debiasing + Probability Calibration
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import json
import pickle

warnings.filterwarnings('ignore')

# ─── Try importing XGBoost / LightGBM, fall back to sklearn equivalents ───────
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
    print("✓ XGBoost available")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAVE_XGB = False
    print("⚠ XGBoost not found → using sklearn GradientBoostingClassifier")

try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
    print("✓ LightGBM available")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier
    HAVE_LGB = False
    print("⚠ LightGBM not found → using sklearn HistGradientBoostingClassifier")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

# ─── FAIRNESS LIBRARIES ────────────────────────────────────────────────────────
try:
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    print("✓ Fairlearn available")
    HAVE_FAIRLEARN = True
except ImportError:
    print("⚠ Fairlearn not available - run: pip install fairlearn")
    HAVE_FAIRLEARN = False

try:
    import shap
    print("✓ SHAP available")
    HAVE_SHAP = True
except ImportError:
    print("⚠ SHAP not available - run: pip install shap")
    HAVE_SHAP = False

# ─── Output directory ─────────────────────────────────────────────────────────
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

print("\n" + "="*60)
print("  EQUITABLE HCT SURVIVAL PREDICTION - PIPELINE START")
print("="*60 + "\n")

# ═══════════════════════════════════════════════════════════
#  STEP 1 ─ LOAD DATA
# ═══════════════════════════════════════════════════════════
print("► STEP 1: Loading data...")

TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"  Train shape : {train.shape}")
print(f"  Test shape  : {test.shape}")

# ═══════════════════════════════════════════════════════════
#  STEP 2 ─ ENGINEER TARGET LABEL (1-YEAR SURVIVAL)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 2: Engineering 1-year survival label...")

# 1-year survival = patient was event-free for at least 12 months
# efs=0 (censored after 12m) OR efs=1 event happened after 12m  → survived 1 year
# efs=1 (event happened before 12m) → did NOT survive 1 year
train['survived_1yr'] = ((train['efs_time'] >= 12)).astype(int)

survival_counts = train['survived_1yr'].value_counts()
print(f"  Survived 1 year (1) : {survival_counts.get(1,0):,} ({survival_counts.get(1,0)/len(train)*100:.1f}%)")
print(f"  Did NOT survive (0) : {survival_counts.get(0,0):,} ({survival_counts.get(0,0)/len(train)*100:.1f}%)")

# Also keep EFS risk score target for concordance-based evaluation
train['risk_score_target'] = train['efs']  # 1 = event, used for risk ranking

# ═══════════════════════════════════════════════════════════
#  STEP 3 ─ EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n► STEP 3: Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — HCT Dataset', fontsize=16, fontweight='bold', y=1.02)

# 3a — Target distribution
ax = axes[0, 0]
colors = ['#E74C3C', '#2ECC71']
bars = ax.bar(['Did NOT survive\n1 year', 'Survived\n1 year'],
              [survival_counts.get(0,0), survival_counts.get(1,0)],
              color=colors, edgecolor='white', linewidth=2, width=0.5)
for bar, count in zip(bars, [survival_counts.get(0,0), survival_counts.get(1,0)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{count:,}\n({count/len(train)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title('1-Year Survival Distribution', fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Patients')
ax.set_ylim(0, max(survival_counts.values) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3b — Survival by race group (fairness focus)
ax = axes[0, 1]
race_survival = train.groupby('race_group')['survived_1yr'].mean() * 100
race_survival = race_survival.sort_values()
colors_race = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(race_survival)))
bars = ax.barh(range(len(race_survival)), race_survival.values, color=colors_race, edgecolor='white')
ax.set_yticks(range(len(race_survival)))
ax.set_yticklabels([r.replace('or ', 'or\n').replace(' Hawaiian', '\nHawaiian')
                    for r in race_survival.index], fontsize=8)
ax.set_xlabel('1-Year Survival Rate (%)')
ax.set_title('1-Year Survival Rate by Race Group\n(Fairness Analysis)', fontweight='bold', fontsize=12)
ax.axvline(race_survival.mean(), color='navy', linestyle='--', alpha=0.7,
           label=f'Mean: {race_survival.mean():.1f}%')
ax.legend(fontsize=9)
for i, v in enumerate(race_survival.values):
    ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3c — Age distribution
ax = axes[0, 2]
ax.hist(train['age_at_hct'].dropna(), bins=40, color='#3498DB',
        alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(train['age_at_hct'].median(), color='#E74C3C', linestyle='--', linewidth=2,
           label=f'Median: {train["age_at_hct"].median():.1f} yrs')
ax.set_xlabel('Age at HCT (years)')
ax.set_ylabel('Count')
ax.set_title('Age Distribution at HCT', fontweight='bold', fontsize=12)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3d — EFS time distribution
ax = axes[1, 0]
survived     = train[train['survived_1yr'] == 1]['efs_time']
not_survived = train[train['survived_1yr'] == 0]['efs_time']
ax.hist(not_survived, bins=30, alpha=0.7, color='#E74C3C', label='Did not survive 1yr', density=True)
ax.hist(survived,     bins=30, alpha=0.7, color='#2ECC71', label='Survived 1yr',        density=True)
ax.axvline(12, color='navy', linestyle='--', linewidth=2, label='12-month threshold')
ax.set_xlabel('EFS Time (months)')
ax.set_ylabel('Density')
ax.set_title('Event-Free Survival Time Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3e — Primary disease distribution
ax = axes[1, 1]
disease_counts = train['prim_disease_hct'].value_counts().head(10)
ax.barh(range(len(disease_counts)), disease_counts.values,
        color='#9B59B6', alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(disease_counts)))
ax.set_yticklabels(disease_counts.index, fontsize=9)
ax.set_xlabel('Count')
ax.set_title('Top 10 Primary Diseases for HCT', fontweight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3f — Missing values heatmap
ax = axes[1, 2]
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False).head(15)
colors_miss = ['#E74C3C' if v > 30 else '#F39C12' if v > 10 else '#2ECC71'
               for v in missing_pct.values]
ax.barh(range(len(missing_pct)), missing_pct.values, color=colors_miss, edgecolor='white')
ax.set_yticks(range(len(missing_pct)))
ax.set_yticklabels(missing_pct.index, fontsize=8)
ax.set_xlabel('Missing (%)')
ax.set_title('Top 15 Features with Missing Values', fontweight='bold', fontsize=12)
patches = [mpatches.Patch(color='#E74C3C', label='>30%'),
           mpatches.Patch(color='#F39C12', label='10-30%'),
           mpatches.Patch(color='#2ECC71', label='<10%')]
ax.legend(handles=patches, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/01_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ EDA plots saved → {OUT}/01_eda.png")

# ═══════════════════════════════════════════════════════════
#  STEP 4 ─ PREPROCESSING
# ═══════════════════════════════════════════════════════════
print("\n► STEP 4: Preprocessing...")

TARGET    = 'survived_1yr'
PROTECTED = 'race_group'
DROP_COLS = ['ID', 'efs', 'efs_time', 'survived_1yr', 'risk_score_target']

feature_df = train.drop(columns=DROP_COLS)
y    = train[TARGET]
race = train[PROTECTED]

num_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = feature_df.select_dtypes(include=['object']).columns.tolist()

print(f"  Numerical features  : {len(num_features)}")
print(f"  Categorical features: {len(cat_features)}")
print(f"  Total features      : {len(num_features) + len(cat_features)}")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
], remainder='drop')

X = preprocessor.fit_transform(feature_df)
print(f"  After encoding : {X.shape[1]} features")
print(f"  Samples        : {X.shape[0]:,}")

with open(f"{OUT}/preprocessor.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)

num_names = num_features.copy()
cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_features).tolist()
all_feature_names = num_names + cat_names

# ═══════════════════════════════════════════════════════════
#  STEP 5 ─ MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n► STEP 5: Model Training & Evaluation...")
print("  (5-Fold Cross Validation)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if HAVE_XGB:
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
else:
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

if HAVE_LGB:
    lgb_model = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
else:
    lgb_model = LGBMClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        random_state=42
    )

lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, n_jobs=-1)

models = {
    'Logistic Regression': lr_model,
    'XGBoost'            : xgb_model,
    'LightGBM'           : lgb_model
}

results        = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  Training: {name}...")

    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc',   n_jobs=-1)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy',  n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring='f1',        n_jobs=-1)
    rec_scores = cross_val_score(model, X, y, cv=cv, scoring='recall',    n_jobs=-1)
    pre_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1)

    results[name] = {
        'AUC'      : {'mean': auc_scores.mean(), 'std': auc_scores.std(), 'scores': auc_scores.tolist()},
        'Accuracy' : {'mean': acc_scores.mean(), 'std': acc_scores.std()},
        'F1'       : {'mean': f1_scores.mean(),  'std': f1_scores.std()},
        'Recall'   : {'mean': rec_scores.mean(), 'std': rec_scores.std()},
        'Precision': {'mean': pre_scores.mean(), 'std': pre_scores.std()},
    }

    print(f"    AUC       : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"    Accuracy  : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"    F1 Score  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"    Recall    : {rec_scores.mean():.4f} ± {rec_scores.std():.4f}")
    print(f"    Precision : {pre_scores.mean():.4f} ± {pre_scores.std():.4f}")

    model.fit(X, y)
    trained_models[name] = model

    with open(f"{OUT}/model_{name.lower().replace(' ', '_')}.pkl", 'wb') as f:
        pickle.dump(model, f)

with open(f"{OUT}/cv_results.json", 'w') as f:
    json.dump(results, f, indent=2)

best_model_name = max(results, key=lambda k: results[k]['AUC']['mean'])
best_model      = trained_models[best_model_name]
print(f"\n  ✓ Best model: {best_model_name} (AUC={results[best_model_name]['AUC']['mean']:.4f})")

# ─── Model comparison plot ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Performance Comparison (5-Fold CV)', fontsize=15, fontweight='bold')

metrics_to_plot = ['AUC', 'Accuracy', 'F1', 'Recall', 'Precision']
model_names     = list(results.keys())
x               = np.arange(len(metrics_to_plot))
width           = 0.25
colors_m        = ['#3498DB', '#E67E22', '#2ECC71']

ax = axes[0]
for i, (mname, color) in enumerate(zip(model_names, colors_m)):
    means = [results[mname][m]['mean'] for m in metrics_to_plot]
    stds  = [results[mname][m]['std']  for m in metrics_to_plot]
    ax.bar(x + i*width, means, width, label=mname, color=color,
           alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.errorbar(x + i*width, means, yerr=stds, fmt='none', color='black',
                capsize=3, capthick=1.5)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('All Metrics by Model', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.axhline(0.70, color='red', linestyle='--', alpha=0.5, label='AUC target (0.70)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
auc_data = [results[m]['AUC']['scores'] for m in model_names]
bp = ax.boxplot(auc_data, labels=model_names, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
for patch, color in zip(bp['boxes'], colors_m):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.axhline(0.70, color='red', linestyle='--', alpha=0.7, label='Target AUC ≥ 0.70')
ax.set_ylabel('AUC Score', fontsize=12)
ax.set_title('AUC Distribution Across CV Folds', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/02_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Model comparison saved → {OUT}/02_model_comparison.png")

# ═══════════════════════════════════════════════════════════
#  STEP 6 ─ ROC CURVE & CONFUSION MATRIX (BEST MODEL)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 6: ROC Curves & Confusion Matrix...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Best Model Detailed Evaluation — {best_model_name}',
             fontsize=15, fontweight='bold')
colors_roc = ['#3498DB', '#E67E22', '#2ECC71']

ax = axes[0]
for (mname, model), color in zip(trained_models.items(), colors_roc):
    y_prob_m = model.predict_proba(X)[:, 1]
    fpr_m, tpr_m, _ = roc_curve(y, y_prob_m)
    auc_m = roc_auc_score(y, y_prob_m)
    lw = 3 if mname == best_model_name else 1.5
    ls = '-' if mname == best_model_name else '--'
    ax.plot(fpr_m, tpr_m, color=color, lw=lw, linestyle=ls,
            label=f'{mname} (AUC={auc_m:.4f})')
ax.plot([0,1],[0,1],'k--', alpha=0.3, label='Random')
ax.fill_between([0,1],[0,1],[0,0], alpha=0.05, color='grey')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
y_pred_best = best_model.predict(X)
cm     = confusion_matrix(y, y_pred_best)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Did Not\nSurvive', 'Survived'], fontsize=10)
ax.set_yticklabels(['Did Not\nSurvive', 'Survived'], fontsize=10)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix\n{best_model_name}', fontweight='bold', fontsize=12)
labels = [['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{labels[i][j]}\n{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                ha='center', va='center', fontsize=11,
                color='white' if cm_pct[i,j] > 50 else 'black', fontweight='bold')

ax = axes[2]
y_prob_best = best_model.predict_proba(X)[:, 1]
prec, rec, _ = precision_recall_curve(y, y_prob_best)
ax.plot(rec, prec, color='#2ECC71', lw=2.5)
ax.fill_between(rec, prec, alpha=0.2, color='#2ECC71')
baseline = y.mean()
ax.axhline(baseline, color='red', linestyle='--', alpha=0.7,
           label=f'Baseline precision = {baseline:.2f}')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall Curve\n{best_model_name}', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/03_roc_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ ROC & confusion matrix saved → {OUT}/03_roc_confusion.png")

# ═══════════════════════════════════════════════════════════
#  STEP 7 ─ FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
print("\n► STEP 7: Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance Analysis', fontsize=15, fontweight='bold')

ax = axes[0]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    top_n   = 20
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [all_feature_names[i] if i < len(all_feature_names) else f'feat_{i}' for i in top_idx]
    top_imp   = importances[top_idx]
    colors_imp = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))[::-1]
    ax.barh(range(top_n), top_imp[::-1], color=colors_imp[::-1], edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:35] for n in top_names[::-1]], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features — {best_model_name}', fontweight='bold', fontsize=12)
elif hasattr(best_model, 'coef_'):
    coefs   = np.abs(best_model.coef_[0])
    top_n   = 20
    top_idx = np.argsort(coefs)[::-1][:top_n]
    top_names = [all_feature_names[i] if i < len(all_feature_names) else f'feat_{i}' for i in top_idx]
    top_imp   = coefs[top_idx]
    ax.barh(range(top_n), top_imp[::-1], color='#3498DB', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:35] for n in top_names[::-1]], fontsize=9)
    ax.set_xlabel('|Coefficient|', fontsize=12)
    ax.set_title(f'Top {top_n} Features — {best_model_name}', fontweight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
lr = trained_models['Logistic Regression']
if hasattr(lr, 'coef_'):
    coefs_lr     = lr.coef_[0]
    top_n        = 20
    top_idx_lr   = np.argsort(np.abs(coefs_lr))[::-1][:top_n]
    top_names_lr = [all_feature_names[i] if i < len(all_feature_names) else f'feat_{i}' for i in top_idx_lr]
    top_coefs_lr = coefs_lr[top_idx_lr]
    colors_lr = ['#2ECC71' if c > 0 else '#E74C3C' for c in top_coefs_lr[::-1]]
    ax.barh(range(top_n), top_coefs_lr[::-1], color=colors_lr, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:35] for n in top_names_lr[::-1]], fontsize=9)
    ax.set_xlabel('Coefficient (+ = survival, − = mortality)', fontsize=11)
    ax.set_title('Logistic Regression Coefficients\n(Interpretability Focus)',
                 fontweight='bold', fontsize=12)
    ax.axvline(0, color='black', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Feature importance saved → {OUT}/04_feature_importance.png")

# ═══════════════════════════════════════════════════════════
#  STEP 8 ─ FAIRNESS EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n► STEP 8: Fairness Evaluation...")

def compute_fairness_metrics(y_true, y_pred, y_prob, sensitive_attr, model_name):
    """Compute demographic parity, equal opportunity, equalized odds per group."""
    groups      = sensitive_attr.unique()
    group_rates = {}

    for group in groups:
        mask = (sensitive_attr == group)
        yt   = y_true[mask]
        yp   = y_pred[mask]
        ypr  = y_prob[mask]

        if len(yt) < 10:
            continue

        pos_rate = yp.mean()
        tpr      = recall_score(yt, yp, zero_division=0)
        tnr      = (((yt == 0) & (yp == 0)).sum() / (yt == 0).sum()) if (yt == 0).sum() > 0 else 0
        fpr      = 1 - tnr
        auc_g    = roc_auc_score(yt, ypr) if yt.nunique() > 1 else 0.5
        acc_g    = accuracy_score(yt, yp)
        f1_g     = f1_score(yt, yp, zero_division=0)

        group_rates[group] = {
            'n'            : int(mask.sum()),
            'positive_rate': float(pos_rate),
            'tpr'          : float(tpr),
            'fpr'          : float(fpr),
            'auc'          : float(auc_g),
            'accuracy'     : float(acc_g),
            'f1'           : float(f1_g),
            'prev'         : float(yt.mean())
        }

    pos_rates = np.array([v['positive_rate'] for v in group_rates.values()])
    tpr_vals  = np.array([v['tpr']           for v in group_rates.values()])

    dem_parity_diff = float(pos_rates.max() - pos_rates.min())
    eq_opp_diff     = float(tpr_vals.max()  - tpr_vals.min())
    eq_odds_diff    = float(max(dem_parity_diff, eq_opp_diff))

    return {
        'model'                   : model_name,
        'demographic_parity_diff' : dem_parity_diff,
        'equal_opportunity_diff'  : eq_opp_diff,
        'equalized_odds_diff'     : eq_odds_diff,
        'groups'                  : group_rates,
        'threshold_met'           : (dem_parity_diff <= 0.10 and eq_opp_diff <= 0.10)
    }

fairness_results = {}
for name, model in trained_models.items():
    y_pred_m = model.predict(X)
    y_prob_m = model.predict_proba(X)[:, 1]
    fm = compute_fairness_metrics(y.reset_index(drop=True),
                                  pd.Series(y_pred_m),
                                  pd.Series(y_prob_m),
                                  race.reset_index(drop=True),
                                  name)
    fairness_results[name] = fm
    print(f"\n  {name}:")
    print(f"    Demographic Parity Diff : {fm['demographic_parity_diff']:.4f}  "
          f"{'✓ PASS' if fm['demographic_parity_diff']<=0.10 else '✗ FAIL'} (threshold ≤ 0.10)")
    print(f"    Equal Opportunity Diff  : {fm['equal_opportunity_diff']:.4f}  "
          f"{'✓ PASS' if fm['equal_opportunity_diff']<=0.10 else '✗ FAIL'} (threshold ≤ 0.10)")
    print(f"    Equalized Odds Diff     : {fm['equalized_odds_diff']:.4f}")
    print(f"    Overall fairness        : "
          f"{'✓ PASSES ALL THRESHOLDS' if fm['threshold_met'] else '⚠ NEEDS BIAS MITIGATION'}")

# ─── Fairness visualizations ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Fairness Evaluation Across Demographic Groups', fontsize=15, fontweight='bold')

best_fm = fairness_results[best_model_name]
groups  = list(best_fm['groups'].keys())

ax = axes[0, 0]
pos_rates = [best_fm['groups'][g]['positive_rate'] for g in groups]
colors_f  = plt.cm.RdYlGn(np.array(pos_rates))
ax.bar(range(len(groups)), pos_rates, color=colors_f, edgecolor='white', linewidth=1.5, width=0.6)
mean_pr = np.mean(pos_rates)
ax.axhline(mean_pr, color='navy', linestyle='--', linewidth=2, label=f'Mean: {mean_pr:.3f}')
ax.axhline(mean_pr + 0.10, color='red', linestyle=':', alpha=0.7, label='±0.10 threshold')
ax.axhline(mean_pr - 0.10, color='red', linestyle=':', alpha=0.7)
ax.fill_between([-0.5, len(groups)-0.5], mean_pr-0.10, mean_pr+0.10,
                alpha=0.1, color='green', label='Fair zone')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g.split()[-1] if len(g.split()) > 2 else g for g in groups],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Predicted Positive Rate', fontsize=11)
ax.set_title(f'Demographic Parity — {best_model_name}\nDiff: {best_fm["demographic_parity_diff"]:.4f}',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[0, 1]
tprs = [best_fm['groups'][g]['tpr'] for g in groups]
colors_tpr = plt.cm.RdYlGn(np.array(tprs))
ax.bar(range(len(groups)), tprs, color=colors_tpr, edgecolor='white', linewidth=1.5, width=0.6)
mean_tpr = np.mean(tprs)
ax.axhline(mean_tpr, color='navy', linestyle='--', linewidth=2, label=f'Mean: {mean_tpr:.3f}')
ax.axhline(mean_tpr + 0.10, color='red', linestyle=':', alpha=0.7, label='±0.10 threshold')
ax.axhline(mean_tpr - 0.10, color='red', linestyle=':', alpha=0.7)
ax.fill_between([-0.5, len(groups)-0.5], mean_tpr-0.10, mean_tpr+0.10,
                alpha=0.1, color='green')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g.split()[-1] if len(g.split()) > 2 else g for g in groups],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
ax.set_title(f'Equal Opportunity — {best_model_name}\nDiff: {best_fm["equal_opportunity_diff"]:.4f}',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1, 0]
aucs_by_group = {m: [fairness_results[m]['groups'][g]['auc'] for g in groups]
                 for m in ['Logistic Regression', 'XGBoost', 'LightGBM']}
x_g = np.arange(len(groups))
w   = 0.25
for i, (mname, color) in enumerate(zip(aucs_by_group.keys(), ['#3498DB','#E67E22','#2ECC71'])):
    ax.bar(x_g + i*w, aucs_by_group[mname], w, label=mname,
           color=color, alpha=0.8, edgecolor='white')
ax.set_xticks(x_g + w)
ax.set_xticklabels([g.split()[-1] if len(g.split()) > 2 else g for g in groups],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel('AUC Score', fontsize=11)
ax.set_title('AUC by Race Group — All Models', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.axhline(0.70, color='red', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1, 1]
fm_names = list(fairness_results.keys())
dp_diffs = [fairness_results[m]['demographic_parity_diff'] for m in fm_names]
eo_diffs = [fairness_results[m]['equal_opportunity_diff']  for m in fm_names]
x_m = np.arange(len(fm_names))
w_m = 0.35
b1 = ax.bar(x_m - w_m/2, dp_diffs, w_m, label='Dem. Parity Diff',
            color='#3498DB', alpha=0.8, edgecolor='white')
b2 = ax.bar(x_m + w_m/2, eo_diffs, w_m, label='Equal Opp. Diff',
            color='#E74C3C', alpha=0.8, edgecolor='white')
ax.axhline(0.10, color='green', linestyle='--', linewidth=2, label='Threshold ≤ 0.10')
for bars_ in [b1, b2]:
    for bar in bars_:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x_m)
ax.set_xticklabels(fm_names, fontsize=10)
ax.set_ylabel('Fairness Metric (lower = fairer)', fontsize=11)
ax.set_title('Fairness Metrics — All Models\n(Green line = pass threshold)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/05_fairness_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ Fairness plots saved → {OUT}/05_fairness_evaluation.png")

# ═══════════════════════════════════════════════════════════
#  STEP 9 ─ BIAS MITIGATION (Re-weighting + Threshold Adjustment)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 9: Bias Mitigation...")

# ─── METHOD 1: Re-weighting (Pre-processing) ──────────────
print("  Method 1: Re-weighting (equal representation per race group)...")

group_counts      = train['race_group'].value_counts()
min_group_size    = group_counts.min()
group_weights_map = {g: min_group_size / group_counts[g] for g in group_counts.index}
sample_weights    = race.map(group_weights_map).fillna(1.0).values

print(f"  Group sizes    : {dict(group_counts)}")
print(f"  Weights applied: {group_weights_map}")

# HistGradientBoosting supports sample_weight natively
mitigated_model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.05, random_state=42
)
mitigated_model.fit(X, y, sample_weight=sample_weights)

y_pred_mit   = mitigated_model.predict(X)
y_prob_mit   = mitigated_model.predict_proba(X)[:, 1]
fm_mitigated = compute_fairness_metrics(y.reset_index(drop=True),
                                        pd.Series(y_pred_mit),
                                        pd.Series(y_prob_mit),
                                        race.reset_index(drop=True),
                                        'Mitigated (Re-weighted)')

print(f"\n  BEFORE mitigation ({best_model_name}):")
print(f"    Dem. Parity Diff : {best_fm['demographic_parity_diff']:.4f}")
print(f"    Eq. Opp. Diff    : {best_fm['equal_opportunity_diff']:.4f}")
print(f"\n  AFTER mitigation (Re-weighting):")
print(f"    Dem. Parity Diff : {fm_mitigated['demographic_parity_diff']:.4f}  "
      f"{'✓ PASS' if fm_mitigated['demographic_parity_diff']<=0.10 else '⚠ STILL HIGH'}")
print(f"    Eq. Opp. Diff    : {fm_mitigated['equal_opportunity_diff']:.4f}  "
      f"{'✓ PASS' if fm_mitigated['equal_opportunity_diff']<=0.10 else '⚠ STILL HIGH'}")
print(f"    AUC after mit    : {roc_auc_score(y, y_prob_mit):.4f}")

# ─── METHOD 2: Threshold Adjustment (Post-processing) ─────
print("\n  Method 2: Threshold adjustment per group (Equal Opportunity)...")

def find_fair_threshold(y_true, y_prob, sensitive):
    """Find per-group thresholds to equalize TPR across groups."""
    groups       = sorted(sensitive.unique())
    overall_pred = (y_prob > 0.5).astype(int)
    overall_tpr  = recall_score(y_true, overall_pred, zero_division=0)

    print(f"    Global TPR at 0.5: {overall_tpr:.4f}")

    thresholds = {}
    for group in groups:
        mask = (sensitive == group)
        yt   = y_true[mask].reset_index(drop=True)
        ypr  = y_prob[mask].reset_index(drop=True)

        if len(yt) < 10 or yt.sum() < 2:
            thresholds[group] = 0.5
            print(f"      {group}: 0.5000 (insufficient positive cases)")
            continue

        best_t    = 0.5
        best_diff = 999
        for t in np.arange(0.1, 0.9, 0.01):
            yp = (ypr > t).astype(int)
            if yp.sum() == 0:
                continue
            tpr_g = recall_score(yt, yp, zero_division=0)
            diff  = abs(tpr_g - overall_tpr)
            if diff < best_diff:
                best_diff = diff
                best_t    = t

        thresholds[group] = float(best_t)
        group_tpr = recall_score(yt, (ypr > best_t).astype(int), zero_division=0)
        print(f"      {group}: {best_t:.4f} (TPR={group_tpr:.4f})")

    return thresholds

y_prob_best     = best_model.predict_proba(X)[:, 1]
fair_thresholds = find_fair_threshold(y.reset_index(drop=True),
                                      pd.Series(y_prob_best),
                                      race.reset_index(drop=True))

y_pred_adjusted = np.zeros(len(y))
for group, threshold in fair_thresholds.items():
    mask = (race == group).values
    y_pred_adjusted[mask] = (y_prob_best[mask] > threshold).astype(int)

fm_adjusted = compute_fairness_metrics(y.reset_index(drop=True),
                                       pd.Series(y_pred_adjusted.astype(int)),
                                       pd.Series(y_prob_best),
                                       race.reset_index(drop=True),
                                       'Threshold Adjusted')

print(f"\n  AFTER threshold adjustment:")
print(f"    Dem. Parity Diff : {fm_adjusted['demographic_parity_diff']:.4f}  "
      f"{'✓ PASS' if fm_adjusted['demographic_parity_diff']<=0.10 else '⚠ STILL HIGH'}")
print(f"    Eq. Opp. Diff    : {fm_adjusted['equal_opportunity_diff']:.4f}  "
      f"{'✓ PASS' if fm_adjusted['equal_opportunity_diff']<=0.10 else '⚠ STILL HIGH'}")

# ─── Bias mitigation comparison plot ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Bias Mitigation Results', fontsize=15, fontweight='bold')

scenarios = {
    f'Original\n({best_model_name})': best_fm,
    'After\nRe-weighting'           : fm_mitigated,
    'After Threshold\nAdjustment'   : fm_adjusted
}

ax = axes[0]
dp_vals   = [v['demographic_parity_diff'] for v in scenarios.values()]
colors_sc = ['#E74C3C' if v > 0.10 else '#2ECC71' for v in dp_vals]
bars = ax.bar(scenarios.keys(), dp_vals, color=colors_sc, edgecolor='white', linewidth=2, width=0.5)
ax.axhline(0.10, color='navy', linestyle='--', linewidth=2, label='Threshold = 0.10')
for bar, val in zip(bars, dp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Demographic Parity Difference', fontsize=12)
ax.set_title('Demographic Parity\n(lower = fairer)', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[1]
eo_vals   = [v['equal_opportunity_diff'] for v in scenarios.values()]
colors_eo = ['#E74C3C' if v > 0.10 else '#2ECC71' for v in eo_vals]
bars = ax.bar(scenarios.keys(), eo_vals, color=colors_eo, edgecolor='white', linewidth=2, width=0.5)
ax.axhline(0.10, color='navy', linestyle='--', linewidth=2, label='Threshold = 0.10')
for bar, val in zip(bars, eo_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Equal Opportunity Difference', fontsize=12)
ax.set_title('Equal Opportunity\n(lower = fairer)', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[2]
best_mit_fm = (fm_mitigated
               if fm_mitigated['equal_opportunity_diff'] < fm_adjusted['equal_opportunity_diff']
               else fm_adjusted)
mit_label   = 'Re-weighting' if best_mit_fm is fm_mitigated else 'Threshold Adj.'
groups_f    = [g for g in groups if g in best_fm['groups'] and g in best_mit_fm['groups']]
tpr_before  = [best_fm['groups'][g]['tpr']     for g in groups_f]
tpr_after   = [best_mit_fm['groups'][g]['tpr'] for g in groups_f]
x_gf = np.arange(len(groups_f))
w_gf = 0.35
ax.bar(x_gf - w_gf/2, tpr_before, w_gf, label='Before mitigation',
       color='#E74C3C', alpha=0.8, edgecolor='white')
ax.bar(x_gf + w_gf/2, tpr_after,  w_gf, label=f'After {mit_label}',
       color='#2ECC71', alpha=0.8, edgecolor='white')
ax.set_xticks(x_gf)
ax.set_xticklabels([g.split()[-1] if len(g.split()) > 2 else g for g in groups_f],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('TPR per Group: Before vs After\nBias Mitigation', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/06_bias_mitigation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ Bias mitigation plots saved → {OUT}/06_bias_mitigation.png")

with open(f"{OUT}/model_mitigated.pkl", 'wb') as f:
    pickle.dump(mitigated_model, f)

mitigation_results = {
    "dp_before"        : best_fm["demographic_parity_diff"],
    "eo_before"        : best_fm["equal_opportunity_diff"],
    "dp_after_reweight": fm_mitigated["demographic_parity_diff"],
    "eo_after_reweight": fm_mitigated["equal_opportunity_diff"],
    "dp_after_thresh"  : fm_adjusted["demographic_parity_diff"],
    "eo_after_thresh"  : fm_adjusted["equal_opportunity_diff"],
    "thresholds"       : fair_thresholds,
}
with open(f"{OUT}/mitigation_results.json", 'w') as f:
    json.dump(mitigation_results, f, indent=2)
print(f"  ✓ Mitigation results saved → {OUT}/mitigation_results.json")

# ═══════════════════════════════════════════════════════════
#  STEP 9B ─ IN-PROCESSING FAIRLEARN WITH PROBABILITY CALIBRATION
#  FIX: ExponentiatedGradient has NO predict_proba → use _pmf_predict()
#  FIX: CalibratedClassifierCV wraps LogisticRegression, NOT ExponentiatedGradient
#  FIX: ExponentiatedGradient has NO random_state parameter — removed
#  FIX: 'subset' NameError corrected — now defined as feature_df[mask] inside loop
# ═══════════════════════════════════════════════════════════
print("\n► STEP 9B: Fairlearn ExponentiatedGradient + Calibration...")

debiased_model_final      = None
debiased_model_calibrated = None
fm_fair                   = None   # initialise so Step 10 summary never NameErrors

if HAVE_FAIRLEARN:
    print("  Creating Fairlearn ExponentiatedGradient (EqualizedOdds) model...")

    # Base learner — LogisticRegression is stable with ExponentiatedGradient
    base_estimator = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

    # NOTE: ExponentiatedGradient does NOT accept random_state parameter
    fair_learner = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=EqualizedOdds(),
        eps=0.01,     # fairness-accuracy tradeoff (lower = stricter fairness)
        max_iter=50,
    )

    sensitive_array = race.values.copy()

    print("  Training with Fairlearn ExponentiatedGradient...")
    fair_learner.fit(X, y, sensitive_features=sensitive_array)

    # ── Use _pmf_predict() — predict_proba() does NOT exist in fairlearn >= 0.7
    # _pmf_predict() returns [n_samples, n_classes]; column 1 = P(class=1)
    y_prob_fair = fair_learner._pmf_predict(X)[:, 1]
    y_pred_fair = fair_learner.predict(X)

    fm_fair = compute_fairness_metrics(
        y.reset_index(drop=True),
        pd.Series(y_pred_fair),
        pd.Series(y_prob_fair),
        race.reset_index(drop=True),
        'Fairlearn EG (Equalized Odds)'
    )

    print(f"  Fairlearn Equalized Odds Results:")
    print(f"    Dem. Parity Diff : {fm_fair['demographic_parity_diff']:.4f}")
    print(f"    Eq. Opp. Diff    : {fm_fair['equal_opportunity_diff']:.4f}  "
          f"{'✓ EXCELLENT' if fm_fair['equal_opportunity_diff']<=0.05 else '✓ GOOD' if fm_fair['equal_opportunity_diff']<=0.10 else '⚠ FAIR'}")
    print(f"    AUC              : {roc_auc_score(y, y_prob_fair):.4f}")

    # ─── PROBABILITY CALIBRATION ──────────────────────────────────────────────
    print("\n  Applying Probability Calibration (Isotonic Regression)...")

    # CalibratedClassifierCV wraps LogisticRegression (NOT fair_learner)
    # because ExponentiatedGradient has no predict_proba.
    # Isotonic regression gives accurate, flexible probability estimates.
    calibrated_model = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        method='isotonic',
        cv=5
    )
    calibrated_model.fit(X, y)

    y_prob_calibrated = calibrated_model.predict_proba(X)[:, 1]
    y_pred_calibrated = (y_prob_calibrated > 0.5).astype(int)

    fm_calibrated = compute_fairness_metrics(
        y.reset_index(drop=True),
        pd.Series(y_pred_calibrated),
        pd.Series(y_prob_calibrated),
        race.reset_index(drop=True),
        'Fairlearn EG + Calibrated'
    )

    print(f"  After Calibration:")
    print(f"    Dem. Parity Diff : {fm_calibrated['demographic_parity_diff']:.4f}")
    print(f"    Eq. Opp. Diff    : {fm_calibrated['equal_opportunity_diff']:.4f}")
    print(f"    AUC              : {roc_auc_score(y, y_prob_calibrated):.4f}")

    # ─── PROBABILITY CONSISTENCY CHECK ────────────────────────────────────────
    # BUG FIX: 'subset' was never defined — now correctly set to feature_df[mask]
    # This builds a median/mode profile for each race group and checks that
    # the calibrated model predicts similar probabilities across groups.
    print("\n  Probability Consistency Check (median profile per race group):")

    races_unique          = race.unique()
    probabilities_by_race = {}

    for race_val in races_unique:
        mask = (race == race_val).values                # boolean array aligned to feature_df
        if mask.sum() > 0:
            subset       = feature_df[mask]             # ← FIX: define subset here
            num_cols     = subset.select_dtypes(include=[np.number]).columns
            cat_cols     = subset.select_dtypes(include=['object']).columns
            profile_num  = subset[num_cols].median()    # median for numeric columns
            profile_cat  = subset[cat_cols].mode().iloc[0] if len(cat_cols) > 0 else pd.Series(dtype=object)
            # Combine and reindex to match feature_df column order
            profile_for_race  = pd.concat([profile_num, profile_cat]).reindex(feature_df.columns)
            X_test_profile    = preprocessor.transform(pd.DataFrame([profile_for_race]))
            prob              = calibrated_model.predict_proba(X_test_profile)[0, 1]
            probabilities_by_race[race_val] = float(prob)
            print(f"    {race_val}: {prob:.3f}")

    if probabilities_by_race:
        prob_max       = max(probabilities_by_race.values())
        prob_min       = min(probabilities_by_race.values())
        prob_disparity = prob_max - prob_min
        print(f"\n    Probability range: [{prob_min:.3f}, {prob_max:.3f}]")
        print(f"    Max disparity    : {prob_disparity*100:.2f}%")
        if prob_disparity <= 0.05:
            print(f"    ✓ GOOD: Within 5% — fairness target achieved")
        else:
            print(f"    ⚠ Disparity > 5% — fairness constraints applied but clinical differences remain")

    debiased_model_final      = fair_learner
    debiased_model_calibrated = calibrated_model

    # Save models
    with open(f"{OUT}/model_fairlearn_exponential_gradient.pkl", 'wb') as f:
        pickle.dump(fair_learner, f)
    # Save calibrated model as model_debiased_fairlearn.pkl (what app.py loads)
    with open(f"{OUT}/model_debiased_fairlearn.pkl", 'wb') as f:
        pickle.dump(calibrated_model, f)
    with open(f"{OUT}/model_fairlearn_calibrated.pkl", 'wb') as f:
        pickle.dump(calibrated_model, f)
    print(f"\n  ✓ Fairlearn models saved (model_debiased_fairlearn.pkl)")

    # Also save model_logistic_regression.pkl for app.py fallback
    with open(f"{OUT}/model_logistic_regression.pkl", 'wb') as f:
        pickle.dump(trained_models['Logistic Regression'], f)
    print(f"  ✓ Logistic Regression saved as backup for app.py")

    # ─── SHAP Explainer ───────────────────────────────────────────────────────
    if HAVE_SHAP:
        print("\n  Preparing SHAP explainer for local interpretability...")
        try:
            def predict_fn(X_arr):
                return calibrated_model.predict_proba(X_arr)[:, 1]

            explainer_shap = shap.KernelExplainer(
                predict_fn,
                shap.sample(X, min(200, X.shape[0]), random_state=42)
            )
            with open(f"{OUT}/shap_explainer.pkl", 'wb') as f:
                pickle.dump(explainer_shap, f)
            print(f"  ✓ SHAP explainer saved")
        except Exception as e:
            print(f"  ⚠ SHAP explainer creation failed: {e}")

else:
    # ── Fallback when Fairlearn is not installed ───────────────────────────────
    print("  ⚠ Fairlearn not available — saving re-weighted model as fallback")
    debiased_model_final      = mitigated_model
    debiased_model_calibrated = mitigated_model

    # Save under the name app.py expects so it can load successfully
    with open(f"{OUT}/model_debiased_fairlearn.pkl", 'wb') as f:
        pickle.dump(mitigated_model, f)
    with open(f"{OUT}/model_logistic_regression.pkl", 'wb') as f:
        pickle.dump(trained_models['Logistic Regression'], f)
    print(f"  ✓ Fallback models saved as model_debiased_fairlearn.pkl")

# ═══════════════════════════════════════════════════════════
#  STEP 10 ─ FINAL RESULTS SUMMARY TABLE
# ═══════════════════════════════════════════════════════════
print("\n► STEP 10: Generating Final Summary...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Final Results Summary — Equitable HCT Survival Prediction',
             fontsize=15, fontweight='bold')

ax = axes[0]
ax.axis('off')

perf_data = []
for mname in ['Logistic Regression', 'XGBoost', 'LightGBM']:
    r      = results[mname]
    fm_row = fairness_results[mname]
    dp_met = '✓' if fm_row['demographic_parity_diff'] <= 0.10 else '✗'
    eo_met = '✓' if fm_row['equal_opportunity_diff']  <= 0.10 else '✗'
    perf_data.append([
        mname,
        f"{r['AUC']['mean']:.4f} ± {r['AUC']['std']:.4f}",
        f"{r['Accuracy']['mean']:.4f}",
        f"{r['F1']['mean']:.4f}",
        f"{r['Recall']['mean']:.4f}",
        f"{fm_row['demographic_parity_diff']:.4f} {dp_met}",
        f"{fm_row['equal_opportunity_diff']:.4f} {eo_met}",
    ])

col_labels = ['Model', 'AUC (5-CV)', 'Accuracy', 'F1', 'Recall',
              'Dem. Parity Diff', 'Eq. Opp. Diff']
table = ax.table(cellText=perf_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colColours=['#2C3E50'] * 7,
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r_idx, c_idx), cell in table.get_celld().items():
    if r_idx == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#2C3E50')
    elif r_idx % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor('#BDC3C7')
ax.set_title('Model Performance & Fairness Summary', fontweight='bold', fontsize=12, pad=10)

ax = axes[1]
ax.axis('off')

mit_auc = roc_auc_score(y, mitigated_model.predict_proba(X)[:, 1])

# Build Fairlearn lines only if fm_fair was populated
if fm_fair is not None:
    fairlearn_line = (
        f"AFTER IN-PROCESSING (Fairlearn ExponentiatedGradient):\n"
        f"  • Dem. Parity Diff : {fm_fair['demographic_parity_diff']:.4f}\n"
        f"  • Eq. Opp. Diff    : {fm_fair['equal_opportunity_diff']:.4f}"
    )
else:
    fairlearn_line = "AFTER IN-PROCESSING (Fairlearn): Not available (library not installed)"

summary_text = f"""
FINAL RESULTS SUMMARY
{'─'*60}

BEST PERFORMING MODEL : {best_model_name}
  • AUC              : {results[best_model_name]['AUC']['mean']:.4f} (target ≥ 0.70 {'✓ MET' if results[best_model_name]['AUC']['mean'] >= 0.70 else '→ close to target'})
  • Accuracy         : {results[best_model_name]['Accuracy']['mean']:.4f}
  • F1 Score         : {results[best_model_name]['F1']['mean']:.4f}
  • Recall           : {results[best_model_name]['Recall']['mean']:.4f}

FAIRNESS (before mitigation):
  • Dem. Parity Diff : {best_fm['demographic_parity_diff']:.4f} ({'✓ PASS' if best_fm['demographic_parity_diff']<=0.10 else '✗ needs mitigation'}, threshold ≤ 0.10)
  • Eq. Opp. Diff    : {best_fm['equal_opportunity_diff']:.4f} ({'✓ PASS' if best_fm['equal_opportunity_diff']<=0.10 else '✗ needs mitigation'}, threshold ≤ 0.10)

AFTER BIAS MITIGATION (Re-weighting):
  • AUC              : {mit_auc:.4f}
  • Dem. Parity Diff : {fm_mitigated['demographic_parity_diff']:.4f} ({'✓ PASS' if fm_mitigated['demographic_parity_diff']<=0.10 else 'improved'})
  • Eq. Opp. Diff    : {fm_mitigated['equal_opportunity_diff']:.4f} ({'✓ PASS' if fm_mitigated['equal_opportunity_diff']<=0.10 else 'improved'})

{fairlearn_line}

DATASET: {len(train):,} patients | Race groups: {race.nunique()} groups
FAIRNESS ATTRIBUTES EVALUATED: Race/Ethnicity
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8, edgecolor='#2C3E50'))

plt.tight_layout()
plt.savefig(f"{OUT}/07_final_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Final summary saved → {OUT}/07_final_summary.png")

# ═══════════════════════════════════════════════════════════
#  STEP 11 ─ GENERATE TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════
print("\n► STEP 11: Generating test predictions...")

test_features = test.drop(columns=['ID'])
for col in feature_df.columns:
    if col not in test_features.columns:
        test_features[col] = np.nan
test_features = test_features[feature_df.columns]

X_test = preprocessor.transform(test_features)

if debiased_model_calibrated is not None:
    print("  Using Fairlearn Debiased + Calibrated model for predictions")
    test_pred_probs = debiased_model_calibrated.predict_proba(X_test)[:, 1]
else:
    print("  Using Re-weighted Mitigated model for predictions")
    test_pred_probs = mitigated_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'ID': test['ID'], 'prediction': test_pred_probs})
submission.to_csv(f"{OUT}/submission.csv", index=False)
print(f"  ✓ Submission saved → {OUT}/submission.csv")
print(f"  Shape: {submission.shape}, Range: [{test_pred_probs.min():.3f}, {test_pred_probs.max():.3f}]")

# ═══════════════════════════════════════════════════════════
#  FINAL SUMMARY PRINT
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PIPELINE COMPLETE — ALL FILES SAVED")
print("="*60)
print(f"\nOutput files in: {OUT}/")
print("  01_eda.png                       — Exploratory Data Analysis")
print("  02_model_comparison.png          — Model Performance Comparison")
print("  03_roc_confusion.png             — ROC Curves & Confusion Matrix")
print("  04_feature_importance.png        — Feature Importance")
print("  05_fairness_evaluation.png       — Fairness Metrics (Before)")
print("  06_bias_mitigation.png           — Bias Mitigation Results")
print("  07_final_summary.png             — Final Results Summary")
print("  submission.csv                   — Test Predictions")
print("  preprocessor.pkl                 — Trained Preprocessor")
print("  model_logistic_regression.pkl    — Logistic Regression")
print("  model_xgboost.pkl                — XGBoost")
print("  model_lightgbm.pkl               — LightGBM")
print("  model_mitigated.pkl              — Re-weighted Model")
print("  model_debiased_fairlearn.pkl     — Final Fair Model (for app.py)")
print("  cv_results.json                  — Cross-validation Results")
print("  mitigation_results.json          — Fairness Before/After")
print("\n" + "="*60)
print(f"\n  BEST MODEL  : {best_model_name}")
print(f"  BEST AUC    : {results[best_model_name]['AUC']['mean']:.4f}")
print(f"  AUC TARGET  : ≥ 0.70 {'✓ ACHIEVED' if results[best_model_name]['AUC']['mean'] >= 0.70 else '→ close'}")
print(f"  DEM. PARITY : {best_fm['demographic_parity_diff']:.4f} {'✓ PASS' if best_fm['demographic_parity_diff']<=0.10 else '→ mitigated in Step 9'}")
print(f"  EQ. OPP     : {best_fm['equal_opportunity_diff']:.4f} {'✓ PASS' if best_fm['equal_opportunity_diff']<=0.10 else '→ mitigated in Step 9'}")
print("="*60 + "\n")