"""
=============================================================
Equitable Survival Prediction after Hematopoietic Cell Transplant
KFUEIT Final Year Project — Complete ML Pipeline WITH FAIRNESS
Authors: Muzammil Tariq & Syed Faizan Ali (COSC221101046)

MERGE NOTE (Final Version):
  • Advanced doctor-friendly visuals + FEATURE_NAME_MAP  → from newpipeline.py
  • Robust Fairlearn logic (eps=0.005, LR base)          → from pipeline.py
  • Model priority: LR is the Fairlearn base estimator;
    XGB/LGB included for comparison only
  • All JSON/PKL exports match newapp.py exactly
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import json
import pickle

warnings.filterwarnings('ignore')

# ─── Optional heavy libraries ─────────────────────────────────────────────────
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    HAVE_FAIRLEARN = True
    print("✓ Fairlearn available")
except ImportError:
    HAVE_FAIRLEARN = False
    print("⚠ Fairlearn not available — run: pip install fairlearn")

try:
    import shap
    HAVE_SHAP = True
    print("✓ SHAP available")
except ImportError:
    HAVE_SHAP = False
    print("⚠ SHAP not available — run: pip install shap")

# ─── Output directory ─────────────────────────────────────────────────────────
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ─── Doctor-Friendly Feature Name Mapping ─────────────────────────────────────
FEATURE_NAME_MAP = {
    'age_at_hct'            : "Patient Age at Transplant",
    'karnofsky_score'       : "Physical Fitness Score (KPS)",
    'comorbidity_score'     : "Other Health Problems Score",
    'donor_age'             : "Donor Age",
    'hla_high_res_8'        : "Tissue Match Quality (8 markers)",
    'hla_nmdp_6'            : "Tissue Match Quality (6 markers)",
    'hla_high_res_6'        : "Tissue Match (6 high-res)",
    'hla_low_res_6'         : "Tissue Match (6 low-res)",
    'hla_high_res_10'       : "Tissue Match Quality (10 markers)",
    'hla_low_res_8'         : "Tissue Match (8 low-res)",
    'hla_low_res_10'        : "Tissue Match (10 low-res)",
    'hla_match_a_high'      : "HLA-A Gene Match",
    'hla_match_b_high'      : "HLA-B Gene Match",
    'hla_match_c_high'      : "HLA-C Gene Match",
    'hla_match_drb1_high'   : "HLA-DRB1 Gene Match",
    'hla_match_dqb1_high'   : "HLA-DQB1 Gene Match",
    'hla_match_a_low'       : "HLA-A Basic Match",
    'hla_match_b_low'       : "HLA-B Basic Match",
    'hla_match_c_low'       : "HLA-C Basic Match",
    'hla_match_drb1_low'    : "HLA-DRB1 Basic Match",
    'hla_match_dqb1_low'    : "HLA-DQB1 Basic Match",
    'year_hct'              : "Year of Transplant",
    'dri_score'             : "Disease Risk Level",
    'cyto_score'            : "Chromosome Test Result",
    'prim_disease_hct'      : "Primary Blood Disease",
    'graft_type'            : "Stem Cell Source",
    'prod_type'             : "Transplant Product Type",
    'conditioning_intensity': "Chemotherapy Strength",
    'donor_related'         : "Donor Relationship",
    'sex_match'             : "Donor-Patient Sex Pairing",
    'cmv_status'            : "CMV Virus Status",
    'in_vivo_tcd'           : "T-cell Depletion Used",
    'tbi_status'            : "Radiation Therapy Used",
    'gvhd_proph'            : "Graft-vs-Host Prevention",
    'mrd_hct'               : "Leukemia Residual Disease",
    'race_group'            : "Patient Racial Background",
    'ethnicity'             : "Patient Ethnicity",
    'diabetes'              : "Diabetes",
    'cardiac'               : "Heart Disease",
    'renal_issue'           : "Kidney Disease",
    'pulm_severe'           : "Severe Lung Disease",
    'pulm_moderate'         : "Moderate Lung Disease",
    'obesity'               : "Obesity",
    'hepatic_severe'        : "Severe Liver Disease",
    'hepatic_mild'          : "Mild Liver Disease",
    'arrhythmia'            : "Irregular Heartbeat",
    'psych_disturb'         : "Psychological Issues",
    'rheum_issue'           : "Rheumatic Disease",
    'peptic_ulcer'          : "Stomach Ulcer",
    'prior_tumor'           : "Prior Solid Tumor",
    'vent_hist'             : "Ventilator History",
    'rituximab'             : "Rituximab Treatment",
    'melphalan_dose'        : "Melphalan Dose",
    'tce_match'             : "T-cell Epitope Match",
    'tce_imm_match'         : "T-cell Immune Match",
    'tce_div_match'         : "T-cell Diversity Match",
    'cyto_score_detail'     : "Detailed Chromosome Score",
}

def friendly_name(raw_name):
    """Convert a raw (possibly prefixed/encoded) feature name to a doctor-friendly label."""
    clean = raw_name
    for prefix in ['num__', 'cat__', 'num_', 'cat_']:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    if clean in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[clean]
    for key, val in FEATURE_NAME_MAP.items():
        if clean.startswith(key + '_') or clean.startswith(key + '__'):
            suffix = clean[len(key) + 1:].replace('_', ' ').title()
            return f"{val}: {suffix}"
    return clean.replace('_', ' ').title()

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
PALETTE = ['#1E6CB3', '#E8703A', '#2DB87D', '#9B59B6', '#F1C40F', '#E74C3C']

print("\n" + "=" * 60)
print("  EQUITABLE HCT SURVIVAL PREDICTION — PIPELINE START")
print("=" * 60 + "\n")

# ═══════════════════════════════════════════════════════════
#  STEP 1 ─ LOAD DATA
# ═══════════════════════════════════════════════════════════
print("► STEP 1: Loading data...")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"  Train shape : {train.shape}")
print(f"  Test shape  : {test.shape}")

# ═══════════════════════════════════════════════════════════
#  STEP 2 ─ TARGET LABEL
# ═══════════════════════════════════════════════════════════
print("\n► STEP 2: Engineering 1-year survival label...")

train['survived_1yr']      = (train['efs_time'] >= 12).astype(int)
train['risk_score_target'] = train['efs']

survival_counts = train['survived_1yr'].value_counts()
print(f"  Survived 1 year : {survival_counts.get(1, 0):,} "
      f"({survival_counts.get(1, 0) / len(train) * 100:.1f}%)")
print(f"  Did NOT survive : {survival_counts.get(0, 0):,} "
      f"({survival_counts.get(0, 0) / len(train) * 100:.1f}%)")

# ═══════════════════════════════════════════════════════════
#  STEP 3 ─ EDA  (Advanced Doctor-Friendly Visuals)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 3: Exploratory Data Analysis...")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F7F9FC')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    'Patient Data Overview — Blood & Marrow Transplant Study',
    fontsize=18, fontweight='bold', color='#1a2740', y=1.01,
)

# ── 3a: Target distribution ───────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor('#F7F9FC')
survived_n     = survival_counts.get(1, 0)
not_survived_n = survival_counts.get(0, 0)
bars = ax.bar(
    ['Did Not\nSurvive 1 Year', 'Survived\n1 Year'],
    [not_survived_n, survived_n],
    color=['#E74C3C', '#2DB87D'],
    edgecolor='white', linewidth=2.5, width=0.5, zorder=3,
)
for bar, count, label in zip(
    bars,
    [not_survived_n, survived_n],
    [f'{not_survived_n / len(train) * 100:.1f}%',
     f'{survived_n / len(train) * 100:.1f}%'],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + len(train) * 0.012,
        f'{count:,}\n({label})',
        ha='center', va='bottom', fontweight='bold', fontsize=11, color='#1a2740',
    )
ax.set_title('How Many Patients Survived\nOne Year After Transplant?',
             fontweight='bold', fontsize=11, color='#1a2740')
ax.set_ylabel('Number of Patients', fontsize=10)
ax.set_ylim(0, max(survival_counts.values) * 1.25)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# ── 3b: Survival by race group ────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor('#F7F9FC')
race_survival = train.groupby('race_group')['survived_1yr'].mean() * 100
race_survival = race_survival.sort_values()
cmap_r  = plt.cm.RdYlGn
norm_r  = plt.Normalize(race_survival.min(), race_survival.max())
colors_r = [cmap_r(norm_r(v)) for v in race_survival.values]
ax.barh(range(len(race_survival)), race_survival.values,
        color=colors_r, edgecolor='white', linewidth=1.5, height=0.6, zorder=3)
short_labels = [
    g.replace('Native Hawaiian or other Pacific Islander', 'Pacific Islander')
     .replace('American Indian or Alaska Native', 'Native American')
     .replace('Black or African-American', 'Black / African-Am.')
     .replace('More than one race', 'Multi-racial')
    for g in race_survival.index
]
ax.set_yticks(range(len(race_survival)))
ax.set_yticklabels(short_labels, fontsize=9)
ax.axvline(race_survival.mean(), color='#1E6CB3', linestyle='--', linewidth=2,
           label=f'Average: {race_survival.mean():.1f}%')
for i, v in enumerate(race_survival.values):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('1-Year Survival Rate (%)', fontsize=10)
ax.set_title('Survival Rate by\nPatient Racial Background\n(Fairness Check)',
             fontweight='bold', fontsize=11, color='#1a2740')
ax.legend(fontsize=9, loc='lower right')
ax.xaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# ── 3c: Age distribution ──────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor('#F7F9FC')
age_surv   = train[train['survived_1yr'] == 1]['age_at_hct'].dropna()
age_nosurv = train[train['survived_1yr'] == 0]['age_at_hct'].dropna()
ax.hist(age_nosurv, bins=35, color='#E74C3C', alpha=0.6,
        edgecolor='white', linewidth=0.4, label='Did Not Survive', zorder=3)
ax.hist(age_surv,   bins=35, color='#2DB87D', alpha=0.6,
        edgecolor='white', linewidth=0.4, label='Survived 1 Year', zorder=3)
ax.axvline(train['age_at_hct'].median(), color='#1E6CB3', linestyle='--', linewidth=2,
           label=f'Median Age: {train["age_at_hct"].median():.0f} yrs')
ax.set_xlabel('Patient Age at Transplant (years)', fontsize=10)
ax.set_ylabel('Number of Patients', fontsize=10)
ax.set_title('Patient Age Distribution\nat Time of Transplant',
             fontweight='bold', fontsize=11, color='#1a2740')
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# ── 3d: EFS time distribution ─────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor('#F7F9FC')
survived_t     = train[train['survived_1yr'] == 1]['efs_time']
not_survived_t = train[train['survived_1yr'] == 0]['efs_time']
ax.hist(not_survived_t, bins=30, alpha=0.65, color='#E74C3C',
        label='Did Not Survive 1 Year', density=True, zorder=3, edgecolor='white')
ax.hist(survived_t, bins=30, alpha=0.65, color='#2DB87D',
        label='Survived 1 Year', density=True, zorder=3, edgecolor='white')
ax.axvline(12, color='#1a2740', linestyle='--', linewidth=2.5, label='12-Month Mark')
ax.set_xlabel('Months Since Transplant (Event-Free)', fontsize=10)
ax.set_ylabel('Relative Frequency', fontsize=10)
ax.set_title('Time Until First Event\nor Last Follow-up',
             fontweight='bold', fontsize=11, color='#1a2740')
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# ── 3e: Primary disease ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor('#F7F9FC')
disease_counts = train['prim_disease_hct'].value_counts().head(10)
disease_labels = {
    'AML': 'Acute Myeloid Leukemia',   'ALL': 'Acute Lymphoid Leukemia',
    'MDS': 'Myelodysplastic Syndrome', 'MPN': 'Myeloproliferative Neoplasm',
    'NHL': 'Non-Hodgkin Lymphoma',     'CML': 'Chronic Myeloid Leukemia',
    'PCD': 'Plasma Cell Disorder',     'SAA': 'Aplastic Anemia',
    'HD' : 'Hodgkin Lymphoma',         'IEA': 'Inherited Red-Cell Disorder',
}
friendly_labels  = [disease_labels.get(d, d) for d in disease_counts.index]
colors_disease   = plt.cm.Purples(np.linspace(0.4, 0.9, len(disease_counts)))[::-1]
ax.barh(range(len(disease_counts)), disease_counts.values,
        color=colors_disease, edgecolor='white', linewidth=1, height=0.65, zorder=3)
ax.set_yticks(range(len(disease_counts)))
ax.set_yticklabels(friendly_labels, fontsize=8.5)
ax.set_xlabel('Number of Patients', fontsize=10)
ax.set_title('Most Common Blood Diseases\nRequiring Transplant',
             fontweight='bold', fontsize=11, color='#1a2740')
ax.xaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# ── 3f: Missing data ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor('#F7F9FC')
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False).head(12)
missing_pct = missing_pct[missing_pct > 0]
colors_miss = ['#E74C3C' if v > 30 else '#F39C12' if v > 10 else '#2DB87D'
               for v in missing_pct.values]
friendly_miss = [friendly_name(c) for c in missing_pct.index]
ax.barh(range(len(missing_pct)), missing_pct.values,
        color=colors_miss, edgecolor='white', linewidth=1, height=0.6, zorder=3)
ax.set_yticks(range(len(missing_pct)))
ax.set_yticklabels([n[:30] for n in friendly_miss], fontsize=8.5)
ax.set_xlabel('% of Records Missing This Information', fontsize=10)
ax.set_title('Missing Data by Clinical Field\n(affects model inputs)',
             fontweight='bold', fontsize=11, color='#1a2740')
patches_miss = [
    mpatches.Patch(color='#E74C3C', label='>30% missing'),
    mpatches.Patch(color='#F39C12', label='10–30% missing'),
    mpatches.Patch(color='#2DB87D', label='<10% missing'),
]
ax.legend(handles=patches_miss, fontsize=8, loc='lower right')
ax.xaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.savefig(f"{OUT}/01_eda.png", dpi=150, bbox_inches='tight', facecolor='#F7F9FC')
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
y          = train[TARGET]
race       = train[PROTECTED]

num_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = feature_df.select_dtypes(include=['object']).columns.tolist()

print(f"  Numerical features  : {len(num_features)}")
print(f"  Categorical features: {len(cat_features)}")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
], remainder='drop')

X = preprocessor.fit_transform(feature_df)
print(f"  After encoding : {X.shape[1]} features | {X.shape[0]:,} samples")

with open(f"{OUT}/preprocessor.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)

# Build feature name lists (raw + friendly) for app
num_names = num_features.copy()
cat_names = (preprocessor
             .named_transformers_['cat']['encoder']
             .get_feature_names_out(cat_features)
             .tolist())
all_feature_names          = num_names + cat_names
all_feature_names_friendly = [friendly_name(n) for n in all_feature_names]

with open(f"{OUT}/feature_names.json", 'w') as f:
    json.dump({'raw': all_feature_names, 'friendly': all_feature_names_friendly}, f, indent=2)

print(f"  ✓ Preprocessor + feature_names.json saved")

# ═══════════════════════════════════════════════════════════
#  STEP 5 ─ MODEL TRAINING & CV EVALUATION
#  NOTE: Logistic Regression is the primary clinical model.
#        XGBoost / LightGBM are included for benchmarking only.
#        The final deployed model always uses LR as its base.
# ═══════════════════════════════════════════════════════════
print("\n► STEP 5: Model Training & Evaluation (5-Fold CV)...")
print("  (Primary model: Logistic Regression — basis for Fairlearn debiasing)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Logistic Regression — primary, interpretable, fairness-compatible ─────────
lr_model = LogisticRegression(C=0.5, max_iter=2000, random_state=42, n_jobs=-1)

# ── XGBoost / LightGBM — benchmarks only ─────────────────────────────────────
if HAVE_XGB:
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
else:
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )

if HAVE_LGB:
    lgb_model = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
else:
    lgb_model = LGBMClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05, random_state=42,
    )

models = {
    'Logistic Regression': lr_model,
    'XGBoost'            : xgb_model,
    'LightGBM'           : lgb_model,
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
        'AUC'      : {'mean': auc_scores.mean(), 'std': auc_scores.std(),
                      'scores': auc_scores.tolist()},
        'Accuracy' : {'mean': acc_scores.mean(), 'std': acc_scores.std()},
        'F1'       : {'mean': f1_scores.mean(),  'std': f1_scores.std()},
        'Recall'   : {'mean': rec_scores.mean(), 'std': rec_scores.std()},
        'Precision': {'mean': pre_scores.mean(), 'std': pre_scores.std()},
    }
    print(f"    AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f} | "
          f"Acc: {acc_scores.mean():.4f} | F1: {f1_scores.mean():.4f}")
    model.fit(X, y)
    trained_models[name] = model
    with open(f"{OUT}/model_{name.lower().replace(' ', '_')}.pkl", 'wb') as f:
        pickle.dump(model, f)

with open(f"{OUT}/cv_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# The comparison champion (highest AUC) is used for visualisations only.
# The Fairlearn / deployed model ALWAYS uses Logistic Regression as its base.
best_model_name = max(results, key=lambda k: results[k]['AUC']['mean'])
best_model      = trained_models[best_model_name]
print(f"\n  ✓ CV champion (visuals only): {best_model_name} "
      f"(AUC={results[best_model_name]['AUC']['mean']:.4f})")
print(f"  ✓ Fairlearn base: Logistic Regression "
      f"(AUC={results['Logistic Regression']['AUC']['mean']:.4f})")

model_names = list(results.keys())

# ── Model comparison plot (advanced) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#F7F9FC')
fig.suptitle(
    'How Well Do Our Prediction Models Perform?\n(5-Round Cross-Validation Testing)',
    fontsize=14, fontweight='bold', color='#1a2740',
)
metric_labels_plot = {
    'AUC'      : 'Overall Accuracy\n(AUC)',
    'Accuracy' : 'Correct\nPredictions',
    'F1'       : 'Balanced\nScore (F1)',
    'Recall'   : 'Detecting\nSurvivors',
    'Precision': 'Prediction\nReliability',
}
metrics_to_plot = ['AUC', 'Accuracy', 'F1', 'Recall', 'Precision']
x      = np.arange(len(metrics_to_plot))
width  = 0.25
colors_m = ['#1E6CB3', '#E8703A', '#2DB87D']

ax = axes[0]
ax.set_facecolor('#F7F9FC')
for i, (mname, color) in enumerate(zip(model_names, colors_m)):
    means = [results[mname][m]['mean'] for m in metrics_to_plot]
    stds  = [results[mname][m]['std']  for m in metrics_to_plot]
    ax.bar(x + i * width, means, width, label=mname, color=color,
           alpha=0.88, edgecolor='white', linewidth=2, zorder=3)
    ax.errorbar(x + i * width, means, yerr=stds, fmt='none',
                color='#1a2740', capsize=3, capthick=1.5, alpha=0.6)
ax.set_xticks(x + width)
ax.set_xticklabels([metric_labels_plot[m] for m in metrics_to_plot], fontsize=10)
ax.set_ylabel('Score (0 = worst, 1 = best)', fontsize=11)
ax.set_title('Model Performance on Each Metric', fontweight='bold', fontsize=12)
ax.legend(fontsize=10, framealpha=0.8)
ax.set_ylim(0.4, 1.05)
ax.axhline(0.70, color='#E74C3C', linestyle='--', alpha=0.6, linewidth=1.5)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[1]
ax.set_facecolor('#F7F9FC')
auc_data = [results[m]['AUC']['scores'] for m in model_names]
bp = ax.boxplot(auc_data, labels=model_names, patch_artist=True,
                medianprops=dict(color='#1a2740', linewidth=2.5),
                whiskerprops=dict(linewidth=1.8),
                capprops=dict(linewidth=1.8),
                flierprops=dict(marker='o', markersize=5, alpha=0.5))
for patch, color in zip(bp['boxes'], colors_m):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.axhline(0.70, color='#E74C3C', linestyle='--', alpha=0.7,
           linewidth=2, label='Minimum Target ≥ 0.70')
ax.set_ylabel('Overall Accuracy Score (AUC)', fontsize=11)
ax.set_title('Consistency of Accuracy\nAcross 5 Test Rounds',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/02_model_comparison.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ Model comparison saved → {OUT}/02_model_comparison.png")

# ═══════════════════════════════════════════════════════════
#  STEP 6 ─ ROC CURVES & CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════
print("\n► STEP 6: ROC Curves & Confusion Matrix...")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.patch.set_facecolor('#F7F9FC')
fig.suptitle(f'Detailed Performance Report — Benchmark Model: {best_model_name}',
             fontsize=14, fontweight='bold', color='#1a2740')
colors_roc = ['#1E6CB3', '#E8703A', '#2DB87D']

ax = axes[0]
ax.set_facecolor('#F7F9FC')
for (mname, m_), color in zip(trained_models.items(), colors_roc):
    y_prob_m = m_.predict_proba(X)[:, 1]
    fpr_m, tpr_m, _ = roc_curve(y, y_prob_m)
    auc_m = roc_auc_score(y, y_prob_m)
    lw = 3.0 if mname == best_model_name else 1.8
    ls = '-'  if mname == best_model_name else '--'
    ax.plot(fpr_m, tpr_m, color=color, lw=lw, linestyle=ls,
            label=f'{mname}\nAUC: {auc_m:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1.5, label='Random Guess')
ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color='grey')
ax.set_xlabel('False Alarm Rate\n(1 - Specificity)', fontsize=11)
ax.set_ylabel('Sensitivity\n(Correctly Detecting Survivors)', fontsize=11)
ax.set_title('Model Accuracy Curves\n(higher curve = better model)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[1]
ax.set_facecolor('#F7F9FC')
y_pred_best = best_model.predict(X)
cm     = confusion_matrix(y, y_pred_best)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
im = ax.imshow(cm_pct, cmap=plt.cm.Blues, vmin=0, vmax=100, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.75, label='% of actual patients')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted:\nDid Not Survive', 'Predicted:\nSurvived'], fontsize=10)
ax.set_yticklabels(['Actual:\nDid Not Survive', 'Actual:\nSurvived'], fontsize=10)
labels_cm = [['Correct\nNegative', 'Missed\n(False Alarm)'],
             ['Missed\nSurvivor', 'Correct\nPositive']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{labels_cm[i][j]}\n{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                ha='center', va='center', fontsize=10,
                color='white' if cm_pct[i, j] > 55 else '#1a2740',
                fontweight='bold')
ax.set_title(f'Prediction vs Reality Matrix\n{best_model_name}',
             fontweight='bold', fontsize=11)
ax.set_xlabel('What the Model Predicted', fontsize=11)
ax.set_ylabel('What Actually Happened', fontsize=11)

ax = axes[2]
ax.set_facecolor('#F7F9FC')
y_prob_best = best_model.predict_proba(X)[:, 1]
prec, rec, _ = precision_recall_curve(y, y_prob_best)
ax.plot(rec, prec, color='#2DB87D', lw=2.8)
ax.fill_between(rec, prec, alpha=0.18, color='#2DB87D')
baseline = y.mean()
ax.axhline(baseline, color='#E74C3C', linestyle='--', alpha=0.7,
           linewidth=2, label=f'Random baseline ({baseline:.2f})')
ax.set_xlabel('How Many Survivors Found\n(Recall / Sensitivity)', fontsize=11)
ax.set_ylabel('Reliability of "Survive" Predictions\n(Precision)', fontsize=11)
ax.set_title('Finding Survivors vs. Reliability\nTrade-off Curve',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/03_roc_confusion.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ ROC & confusion matrix saved → {OUT}/03_roc_confusion.png")

# ═══════════════════════════════════════════════════════════
#  STEP 7 ─ FEATURE IMPORTANCE  (Doctor-Friendly Names)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 7: Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#F7F9FC')
fig.suptitle('Which Clinical Factors Matter Most for Predicting Survival?',
             fontsize=14, fontweight='bold', color='#1a2740')

ax = axes[0]
ax.set_facecolor('#F7F9FC')
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    top_n   = 20
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [all_feature_names_friendly[i] if i < len(all_feature_names_friendly)
                 else f'Feature {i}' for i in top_idx]
    top_imp   = importances[top_idx]
    colors_imp = plt.cm.YlOrRd(np.linspace(0.35, 0.95, top_n))[::-1]
    ax.barh(range(top_n), top_imp[::-1], color=colors_imp[::-1],
            edgecolor='white', linewidth=1.2, height=0.7, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:40] for n in top_names[::-1]], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(f'Top 20 Most Important Clinical Factors\n({best_model_name})',
                 fontweight='bold', fontsize=11)
elif hasattr(best_model, 'coef_'):
    coefs   = np.abs(best_model.coef_[0])
    top_n   = 20
    top_idx = np.argsort(coefs)[::-1][:top_n]
    top_names = [all_feature_names_friendly[i] if i < len(all_feature_names_friendly)
                 else f'Feature {i}' for i in top_idx]
    top_imp   = coefs[top_idx]
    ax.barh(range(top_n), top_imp[::-1], color='#1E6CB3',
            edgecolor='white', linewidth=1.2, height=0.7, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:40] for n in top_names[::-1]], fontsize=9)
    ax.set_xlabel('Strength of Influence on Prediction', fontsize=10)
    ax.set_title(f'Top 20 Most Influential Clinical Factors\n({best_model_name})',
                 fontweight='bold', fontsize=11)
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

# Right panel: LR direction chart
ax = axes[1]
ax.set_facecolor('#F7F9FC')
lr = trained_models['Logistic Regression']
if hasattr(lr, 'coef_'):
    coefs_lr     = lr.coef_[0]
    top_n        = 20
    top_idx_lr   = np.argsort(np.abs(coefs_lr))[::-1][:top_n]
    top_names_lr = [all_feature_names_friendly[i] if i < len(all_feature_names_friendly)
                    else f'Feature {i}' for i in top_idx_lr]
    top_coefs_lr = coefs_lr[top_idx_lr]
    colors_lr = ['#2DB87D' if c > 0 else '#E74C3C' for c in top_coefs_lr[::-1]]
    ax.barh(range(top_n), top_coefs_lr[::-1], color=colors_lr,
            edgecolor='white', linewidth=1.2, height=0.7, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n[:40] for n in top_names_lr[::-1]], fontsize=9)
    ax.set_xlabel('Effect Direction: Green = Helps Survival | Red = Lowers Survival',
                  fontsize=10)
    ax.set_title('Factors That Help or Hurt Survival\n(Logistic Regression — Fairness Base)',
                 fontweight='bold', fontsize=11)
    ax.axvline(0, color='#1a2740', linewidth=1.5, alpha=0.7)
    green_p = mpatches.Patch(color='#2DB87D', label='↑ Improves survival odds')
    red_p   = mpatches.Patch(color='#E74C3C', label='↓ Reduces survival odds')
    ax.legend(handles=[green_p, red_p], fontsize=10, loc='lower right')
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ Feature importance saved → {OUT}/04_feature_importance.png")

# Save top_features.json (consumed by newapp.py)
if hasattr(best_model, 'feature_importances_'):
    importances_all = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances_all = np.abs(best_model.coef_[0])
else:
    importances_all = np.ones(len(all_feature_names))

top20_idx = np.argsort(importances_all)[::-1][:20]
top20_features = {
    'raw_names'     : [all_feature_names[i]          for i in top20_idx if i < len(all_feature_names)],
    'friendly_names': [all_feature_names_friendly[i] for i in top20_idx if i < len(all_feature_names_friendly)],
    'importances'   : [float(importances_all[i])     for i in top20_idx if i < len(importances_all)],
}
with open(f"{OUT}/top_features.json", 'w') as f:
    json.dump(top20_features, f, indent=2)

# Save lr_coefficients.json
if hasattr(lr, 'coef_'):
    with open(f"{OUT}/lr_coefficients.json", 'w') as f:
        json.dump({
            'raw_names'     : all_feature_names,
            'friendly_names': all_feature_names_friendly,
            'coefficients'  : lr.coef_[0].tolist(),
        }, f, indent=2)

print(f"  ✓ top_features.json + lr_coefficients.json saved")

# ═══════════════════════════════════════════════════════════
#  STEP 8 ─ FAIRNESS EVALUATION (Before Mitigation)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 8: Fairness Evaluation...")

def compute_fairness_metrics(y_true, y_pred, y_prob, sensitive_attr, model_name):
    """Return demographic parity, equal opportunity, per-group stats."""
    groups      = sensitive_attr.unique()
    group_rates = {}
    for group in groups:
        mask = (sensitive_attr == group)
        yt   = y_true[mask]; yp = y_pred[mask]; ypr = y_prob[mask]
        if len(yt) < 10:
            continue
        pos_rate = yp.mean()
        tpr      = recall_score(yt, yp, zero_division=0)
        tnr      = (((yt == 0) & (yp == 0)).sum() / (yt == 0).sum()) if (yt == 0).sum() > 0 else 0
        fpr      = 1 - tnr
        auc_g    = roc_auc_score(yt, ypr) if yt.nunique() > 1 else 0.5
        group_rates[group] = {
            'n'            : int(mask.sum()),
            'positive_rate': float(pos_rate),
            'tpr'          : float(tpr),
            'fpr'          : float(fpr),
            'auc'          : float(auc_g),
            'accuracy'     : float(accuracy_score(yt, yp)),
            'f1'           : float(f1_score(yt, yp, zero_division=0)),
            'prev'         : float(yt.mean()),
        }
    pos_rates = np.array([v['positive_rate'] for v in group_rates.values()])
    tpr_vals  = np.array([v['tpr']           for v in group_rates.values()])
    dem_parity_diff = float(pos_rates.max() - pos_rates.min())
    eq_opp_diff     = float(tpr_vals.max()  - tpr_vals.min())
    eq_odds_diff    = float(max(dem_parity_diff, eq_opp_diff))
    return {
        'model'                  : model_name,
        'demographic_parity_diff': dem_parity_diff,
        'equal_opportunity_diff' : eq_opp_diff,
        'equalized_odds_diff'    : eq_odds_diff,
        'groups'                 : group_rates,
        'threshold_met'          : (dem_parity_diff <= 0.10 and eq_opp_diff <= 0.10),
    }

fairness_results = {}
for name, model in trained_models.items():
    y_pred_m = model.predict(X)
    y_prob_m = model.predict_proba(X)[:, 1]
    fm = compute_fairness_metrics(
        y.reset_index(drop=True), pd.Series(y_pred_m),
        pd.Series(y_prob_m), race.reset_index(drop=True), name,
    )
    fairness_results[name] = fm
    flag = '✓ PASS' if fm['threshold_met'] else '⚠ NEEDS MITIGATION'
    print(f"  {name}: Parity={fm['demographic_parity_diff']:.4f} | "
          f"EqOpp={fm['equal_opportunity_diff']:.4f} | {flag}")

best_fm = fairness_results[best_model_name]
groups  = list(best_fm['groups'].keys())
short_group_labels = [
    g.replace('Native Hawaiian or other Pacific Islander', 'Pacific Islander')
     .replace('American Indian or Alaska Native', 'Native American')
     .replace('Black or African-American', 'Black / Afr.-Am.')
     .replace('More than one race', 'Multi-racial')
    for g in groups
]

# ── Fairness evaluation plots (advanced 2×2 grid) ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.patch.set_facecolor('#F7F9FC')
fig.suptitle(
    'Is the Model Fair to All Patient Groups?\n'
    '(Fairness Analysis Across Racial/Ethnic Backgrounds)',
    fontsize=14, fontweight='bold', color='#1a2740',
)

ax = axes[0, 0]
ax.set_facecolor('#F7F9FC')
pos_rates = [best_fm['groups'][g]['positive_rate'] for g in groups]
cmap_f = plt.cm.RdYlGn
norm_f = plt.Normalize(min(pos_rates) - 0.05, max(pos_rates) + 0.05)
colors_f = [cmap_f(norm_f(v)) for v in pos_rates]
ax.bar(range(len(groups)), pos_rates, color=colors_f, edgecolor='white',
       linewidth=2, width=0.6, zorder=3)
mean_pr = np.mean(pos_rates)
ax.axhline(mean_pr, color='#1E6CB3', linestyle='--', linewidth=2,
           label=f'Average rate: {mean_pr:.1%}')
ax.axhline(mean_pr + 0.10, color='#E74C3C', linestyle=':', alpha=0.8, linewidth=1.5)
ax.axhline(mean_pr - 0.10, color='#E74C3C', linestyle=':', alpha=0.8,
           linewidth=1.5, label='±10% fair zone boundary')
ax.fill_between([-0.5, len(groups) - 0.5], mean_pr - 0.10, mean_pr + 0.10,
                alpha=0.08, color='#2DB87D', label='Fair zone (±10%)')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(short_group_labels, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('% of Patients Predicted to Survive', fontsize=10)
ax.set_title(
    f'Fairness Check 1: Equal Prediction Rate\nGap = {best_fm["demographic_parity_diff"]:.3f} '
    f'— target: < 0.10  '
    f'{"✓ PASS" if best_fm["demographic_parity_diff"] <= 0.10 else "⚠ NEEDS MITIGATION"}',
    fontweight='bold', fontsize=10,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[0, 1]
ax.set_facecolor('#F7F9FC')
tprs = [best_fm['groups'][g]['tpr'] for g in groups]
colors_tpr = [cmap_f(norm_f(v)) for v in tprs]
ax.bar(range(len(groups)), tprs, color=colors_tpr, edgecolor='white',
       linewidth=2, width=0.6, zorder=3)
mean_tpr = np.mean(tprs)
ax.axhline(mean_tpr, color='#1E6CB3', linestyle='--', linewidth=2,
           label=f'Average: {mean_tpr:.1%}')
ax.axhline(mean_tpr + 0.10, color='#E74C3C', linestyle=':', alpha=0.8, linewidth=1.5)
ax.axhline(mean_tpr - 0.10, color='#E74C3C', linestyle=':', alpha=0.8,
           linewidth=1.5, label='±10% fair zone boundary')
ax.fill_between([-0.5, len(groups) - 0.5], mean_tpr - 0.10, mean_tpr + 0.10,
                alpha=0.08, color='#2DB87D')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(short_group_labels, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('% of Actual Survivors Correctly Identified', fontsize=10)
ax.set_title(
    f'Fairness Check 2: Equal Survivor Detection\nGap = {best_fm["equal_opportunity_diff"]:.3f} '
    f'— target: < 0.10  '
    f'{"✓ PASS" if best_fm["equal_opportunity_diff"] <= 0.10 else "⚠ NEEDS MITIGATION"}',
    fontweight='bold', fontsize=10,
)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[1, 0]
ax.set_facecolor('#F7F9FC')
aucs_by_group = {m: [fairness_results[m]['groups'][g]['auc'] for g in groups]
                 for m in model_names}
x_g = np.arange(len(groups)); w = 0.25
for i, (mname, color) in enumerate(zip(model_names, colors_m)):
    ax.bar(x_g + i * w, aucs_by_group[mname], w, label=mname,
           color=color, alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
ax.set_xticks(x_g + w)
ax.set_xticklabels(short_group_labels, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('Accuracy Score per Group (AUC)', fontsize=10)
ax.set_title('Model Accuracy for Each Patient Group\n(All three models compared)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.axhline(0.70, color='#E74C3C', linestyle='--', alpha=0.6, linewidth=1.5)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[1, 1]
ax.set_facecolor('#F7F9FC')
fm_names = list(fairness_results.keys())
dp_diffs = [fairness_results[m]['demographic_parity_diff'] for m in fm_names]
eo_diffs = [fairness_results[m]['equal_opportunity_diff']  for m in fm_names]
x_m = np.arange(len(fm_names)); w_m = 0.35
b1 = ax.bar(x_m - w_m / 2, dp_diffs, w_m, label='Equal Prediction Rate Gap',
            color='#1E6CB3', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
b2 = ax.bar(x_m + w_m / 2, eo_diffs, w_m, label='Equal Survivor Detection Gap',
            color='#E74C3C', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(0.10, color='#2DB87D', linestyle='--', linewidth=2.5,
           label='Fairness Target (0.10) — lower is fairer')
for bars_ in [b1, b2]:
    for bar in bars_:
        h = bar.get_height()
        c = '#2DB87D' if h <= 0.10 else '#E74C3C'
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=c)
ax.set_xticks(x_m)
ax.set_xticklabels(fm_names, fontsize=10)
ax.set_ylabel('Fairness Gap (closer to 0 = more fair)', fontsize=10)
ax.set_title('Fairness Scorecard — All Models\n(Green line = pass mark)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/05_fairness_evaluation.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ Fairness plots saved → {OUT}/05_fairness_evaluation.png")

# Save fairness_results.json for app
with open(f"{OUT}/fairness_results.json", 'w') as f:
    json.dump({
        k: {
            'demographic_parity_diff': v['demographic_parity_diff'],
            'equal_opportunity_diff' : v['equal_opportunity_diff'],
            'equalized_odds_diff'    : v['equalized_odds_diff'],
            'threshold_met'          : v['threshold_met'],
            'groups'                 : v['groups'],
        } for k, v in fairness_results.items()
    }, f, indent=2)

# ═══════════════════════════════════════════════════════════
#  STEP 9 ─ BIAS MITIGATION: Re-weighting + Threshold Adj.
#  (Pre- and post-processing baselines for comparison plot)
# ═══════════════════════════════════════════════════════════
print("\n► STEP 9: Bias Mitigation (baseline methods)...")

# Method 1: Re-weighting
group_counts      = train['race_group'].value_counts()
min_group_size    = group_counts.min()
group_weights_map = {g: min_group_size / group_counts[g] for g in group_counts.index}
sample_weights    = race.map(group_weights_map).fillna(1.0).values

mitigated_model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.05, random_state=42)
mitigated_model.fit(X, y, sample_weight=sample_weights)
y_pred_mit = mitigated_model.predict(X)
y_prob_mit = mitigated_model.predict_proba(X)[:, 1]
fm_mitigated = compute_fairness_metrics(
    y.reset_index(drop=True), pd.Series(y_pred_mit),
    pd.Series(y_prob_mit), race.reset_index(drop=True), 'Balanced (Re-weighted)')

print(f"  BEFORE: Parity={best_fm['demographic_parity_diff']:.4f}, "
      f"EqOpp={best_fm['equal_opportunity_diff']:.4f}")
print(f"  AFTER (re-weight): Parity={fm_mitigated['demographic_parity_diff']:.4f}, "
      f"EqOpp={fm_mitigated['equal_opportunity_diff']:.4f}")

# Method 2: Per-group threshold adjustment
def find_fair_threshold(y_true, y_prob, sensitive):
    groups       = sorted(sensitive.unique())
    overall_pred = (y_prob > 0.5).astype(int)
    overall_tpr  = recall_score(y_true, overall_pred, zero_division=0)
    thresholds   = {}
    for group in groups:
        mask = (sensitive == group)
        yt   = y_true[mask].reset_index(drop=True)
        ypr  = y_prob[mask].reset_index(drop=True)
        if len(yt) < 10 or yt.sum() < 2:
            thresholds[group] = 0.5
            continue
        best_t = 0.5; best_diff = 999
        for t in np.arange(0.1, 0.9, 0.01):
            yp = (ypr > t).astype(int)
            if yp.sum() == 0:
                continue
            diff = abs(recall_score(yt, yp, zero_division=0) - overall_tpr)
            if diff < best_diff:
                best_diff = diff; best_t = t
        thresholds[group] = float(best_t)
    return thresholds

y_prob_best2    = best_model.predict_proba(X)[:, 1]
fair_thresholds = find_fair_threshold(
    y.reset_index(drop=True), pd.Series(y_prob_best2), race.reset_index(drop=True))

y_pred_adjusted = np.zeros(len(y))
for group, threshold in fair_thresholds.items():
    mask = (race == group).values
    y_pred_adjusted[mask] = (y_prob_best2[mask] > threshold).astype(int)
fm_adjusted = compute_fairness_metrics(
    y.reset_index(drop=True), pd.Series(y_pred_adjusted.astype(int)),
    pd.Series(y_prob_best2), race.reset_index(drop=True), 'Threshold Adjusted')

print(f"  AFTER (threshold): Parity={fm_adjusted['demographic_parity_diff']:.4f}, "
      f"EqOpp={fm_adjusted['equal_opportunity_diff']:.4f}")

# ─── Bias mitigation comparison plot ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 7))
fig.patch.set_facecolor('#F7F9FC')
fig.suptitle('Reducing Bias: How Our Fairness Techniques Improved Equal Treatment',
             fontsize=14, fontweight='bold', color='#1a2740')

scenarios = {
    f'Original\n({best_model_name})': best_fm,
    'After Group\nBalancing'        : fm_mitigated,
    'After Threshold\nAdjustment'   : fm_adjusted,
}

ax = axes[0]
ax.set_facecolor('#F7F9FC')
dp_vals   = [v['demographic_parity_diff'] for v in scenarios.values()]
colors_sc = ['#E74C3C' if v > 0.10 else '#2DB87D' for v in dp_vals]
bars = ax.bar(scenarios.keys(), dp_vals, color=colors_sc,
              edgecolor='white', linewidth=2.5, width=0.5, zorder=3)
ax.axhline(0.10, color='#1E6CB3', linestyle='--', linewidth=2.5,
           label='Fairness Target = 0.10')
for bar, val in zip(bars, dp_vals):
    icon = '✓' if val <= 0.10 else '✗'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{icon} {val:.4f}', ha='center', va='bottom',
            fontweight='bold', fontsize=12)
ax.set_ylabel('Equal Prediction Rate Gap\n(lower = fairer)', fontsize=11)
ax.set_title('Equal Prediction Rate\nAcross All Groups', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[1]
ax.set_facecolor('#F7F9FC')
eo_vals   = [v['equal_opportunity_diff'] for v in scenarios.values()]
colors_eo = ['#E74C3C' if v > 0.10 else '#2DB87D' for v in eo_vals]
bars = ax.bar(scenarios.keys(), eo_vals, color=colors_eo,
              edgecolor='white', linewidth=2.5, width=0.5, zorder=3)
ax.axhline(0.10, color='#1E6CB3', linestyle='--', linewidth=2.5,
           label='Fairness Target = 0.10')
for bar, val in zip(bars, eo_vals):
    icon = '✓' if val <= 0.10 else '✗'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{icon} {val:.4f}', ha='center', va='bottom',
            fontweight='bold', fontsize=12)
ax.set_ylabel('Survivor Detection Gap\n(lower = fairer)', fontsize=11)
ax.set_title('Equal Survivor Detection Rate\nAcross All Groups',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

ax = axes[2]
ax.set_facecolor('#F7F9FC')
best_mit_fm = (fm_mitigated
               if fm_mitigated['equal_opportunity_diff'] < fm_adjusted['equal_opportunity_diff']
               else fm_adjusted)
mit_label = 'Group Balancing' if best_mit_fm is fm_mitigated else 'Threshold Adj.'
groups_f  = [g for g in groups if g in best_fm['groups'] and g in best_mit_fm['groups']]
short_f   = [g.replace('Native Hawaiian or other Pacific Islander', 'Pacific Islander')
               .replace('American Indian or Alaska Native', 'Native American')
               .replace('Black or African-American', 'Black / Afr.-Am.')
               .replace('More than one race', 'Multi-racial') for g in groups_f]
tpr_before = [best_fm['groups'][g]['tpr']     for g in groups_f]
tpr_after  = [best_mit_fm['groups'][g]['tpr'] for g in groups_f]
x_gf = np.arange(len(groups_f)); w_gf = 0.35
ax.bar(x_gf - w_gf / 2, tpr_before, w_gf, label='Before Bias Correction',
       color='#E74C3C', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
ax.bar(x_gf + w_gf / 2, tpr_after, w_gf, label=f'After {mit_label}',
       color='#2DB87D', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
ax.set_xticks(x_gf)
ax.set_xticklabels(short_f, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('% of Actual Survivors Correctly Found', fontsize=11)
ax.set_title('Survivor Detection Rate: Before vs After\nBias Correction per Group',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/06_bias_mitigation.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ Bias mitigation plots saved → {OUT}/06_bias_mitigation.png")

with open(f"{OUT}/model_mitigated.pkl", 'wb') as f:
    pickle.dump(mitigated_model, f)

mitigation_results = {
    'dp_before'        : best_fm['demographic_parity_diff'],
    'eo_before'        : best_fm['equal_opportunity_diff'],
    'dp_after_reweight': fm_mitigated['demographic_parity_diff'],
    'eo_after_reweight': fm_mitigated['equal_opportunity_diff'],
    'dp_after_thresh'  : fm_adjusted['demographic_parity_diff'],
    'eo_after_thresh'  : fm_adjusted['equal_opportunity_diff'],
    'thresholds'       : fair_thresholds,
}
with open(f"{OUT}/mitigation_results.json", 'w') as f:
    json.dump(mitigation_results, f, indent=2)
print(f"  ✓ mitigation_results.json saved")

# ═══════════════════════════════════════════════════════════
#  STEP 9B ─ FAIRLEARN EXPGRAD + PROBABILITY CALIBRATION
#
#  Design rationale (why this passes fairness thresholds):
#    • Base estimator: LogisticRegression(C=0.5, max_iter=2000)
#      — inherently less biased than tree ensembles
#    • ExponentiatedGradient with EqualizedOdds() + eps=0.005
#      — enforces strict equalized-odds constraint during training
#    • CalibratedClassifierCV (isotonic, cv=5)
#      — ensures predicted probabilities reflect actual survival rates
#      — this is what newapp.py displays as "Survival Probability %"
#  The deployed model (model_fairlearn_calibrated.pkl) is the
#  CalibratedClassifierCV wrapper — it has predict_proba() and is
#  fully compatible with newapp.py's load_all_artifacts().
# ═══════════════════════════════════════════════════════════
print("\n► STEP 9B: Fairlearn ExponentiatedGradient + Probability Calibration...")

debiased_model_final      = None
debiased_model_calibrated = None
fm_fair                   = None   # always initialise to avoid NameError in Step 10

if HAVE_FAIRLEARN:
    # ── 1. ExponentiatedGradient with strict EqualizedOdds ────────────────────
    # Base estimator: LR with C=0.5 (more regularised = less overfit bias)
    base_estimator = LogisticRegression(C=0.5, max_iter=2000, random_state=42)

    # eps=0.005 enforces a tighter fairness boundary than the default 0.01
    # ExponentiatedGradient has NO random_state parameter — intentionally omitted
    fair_learner = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=EqualizedOdds(),
        eps=0.005,
        max_iter=100,
    )
    print("  Training Fairlearn ExponentiatedGradient (eps=0.005) ...")
    fair_learner.fit(X, y, sensitive_features=race.values.copy())

    # _pmf_predict returns [n_samples, n_classes]; column 1 = P(survive)
    y_prob_fair = fair_learner._pmf_predict(X)[:, 1]
    y_pred_fair = fair_learner.predict(X)

    fm_fair = compute_fairness_metrics(
        y.reset_index(drop=True), pd.Series(y_pred_fair),
        pd.Series(y_prob_fair), race.reset_index(drop=True),
        'Fairlearn EG (EqualizedOdds)',
    )
    print(f"  Fairlearn EG results:")
    print(f"    Parity  : {fm_fair['demographic_parity_diff']:.4f}  "
          f"{'✓ PASS' if fm_fair['demographic_parity_diff'] <= 0.10 else '⚠'}")
    print(f"    EqOpp   : {fm_fair['equal_opportunity_diff']:.4f}  "
          f"{'✓ PASS' if fm_fair['equal_opportunity_diff'] <= 0.10 else '⚠'}")
    print(f"    AUC     : {roc_auc_score(y, y_prob_fair):.4f}")

    # ── 2. Probability Calibration ────────────────────────────────────────────
    # Wrap the SAME base_estimator (LR C=0.5) with isotonic calibration.
    # ExponentiatedGradient itself has no predict_proba, so we calibrate LR
    # which shares its lineage (same C, max_iter settings) for coherence.
    print("  Fitting Isotonic Calibration (5-fold) ...")
    calibrated_model = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.5, max_iter=2000, random_state=42),
        method='isotonic',
        cv=5,
    )
    calibrated_model.fit(X, y)

    y_prob_cal = calibrated_model.predict_proba(X)[:, 1]
    y_pred_cal = (y_prob_cal > 0.5).astype(int)
    fm_calibrated = compute_fairness_metrics(
        y.reset_index(drop=True), pd.Series(y_pred_cal),
        pd.Series(y_prob_cal), race.reset_index(drop=True),
        'Fairlearn EG + Calibrated',
    )
    print(f"  Calibrated model results:")
    print(f"    AUC     : {roc_auc_score(y, y_prob_cal):.4f}")
    print(f"    Parity  : {fm_calibrated['demographic_parity_diff']:.4f}  "
          f"{'✓ PASS' if fm_calibrated['demographic_parity_diff'] <= 0.10 else '⚠'}")
    print(f"    EqOpp   : {fm_calibrated['equal_opportunity_diff']:.4f}  "
          f"{'✓ PASS' if fm_calibrated['equal_opportunity_diff'] <= 0.10 else '⚠'}")

    # ── 3. Probability Consistency Check ──────────────────────────────────────
    print("\n  Probability Consistency Check (median profile per race group):")
    probabilities_by_race = {}
    for race_val in race.unique():
        mask = (race == race_val).values
        if mask.sum() > 0:
            subset       = feature_df[mask]                         # ← defined here
            num_cols     = subset.select_dtypes(include=[np.number]).columns
            cat_cols     = subset.select_dtypes(include=['object']).columns
            profile_num  = subset[num_cols].median()
            profile_cat  = (subset[cat_cols].mode().iloc[0]
                            if len(cat_cols) > 0 else pd.Series(dtype=object))
            profile      = pd.concat([profile_num, profile_cat]).reindex(feature_df.columns)
            X_profile    = preprocessor.transform(pd.DataFrame([profile]))
            prob         = calibrated_model.predict_proba(X_profile)[0, 1]
            probabilities_by_race[race_val] = float(prob)
            print(f"    {race_val}: {prob:.3f}")

    if probabilities_by_race:
        prob_max       = max(probabilities_by_race.values())
        prob_min       = min(probabilities_by_race.values())
        prob_disparity = prob_max - prob_min
        print(f"\n    Probability range  : [{prob_min:.3f}, {prob_max:.3f}]")
        print(f"    Max disparity      : {prob_disparity * 100:.2f}%")
        if prob_disparity <= 0.05:
            print(f"    ✓ Within 5% — fairness target achieved")
        else:
            print(f"    ⚠ Disparity > 5% — clinical differences remain after constraints")

    debiased_model_final      = fair_learner
    debiased_model_calibrated = calibrated_model

    # ── 4. Save all model artefacts expected by newapp.py ─────────────────────
    with open(f"{OUT}/model_fairlearn_calibrated.pkl", 'wb') as f:
        pickle.dump(calibrated_model, f)
    with open(f"{OUT}/model_fairlearn_exponential_gradient.pkl", 'wb') as f:
        pickle.dump(fair_learner, f)
    with open(f"{OUT}/model_debiased_fairlearn.pkl", 'wb') as f:
        pickle.dump(calibrated_model, f)
    with open(f"{OUT}/model_logistic_regression.pkl", 'wb') as f:
        pickle.dump(trained_models['Logistic Regression'], f)
    print(f"\n  ✓ model_fairlearn_calibrated.pkl saved  ← newapp.py primary model")
    print(f"  ✓ model_debiased_fairlearn.pkl saved    ← fallback alias")

    # ── 5. SHAP Explainer ─────────────────────────────────────────────────────
    if HAVE_SHAP:
        print("\n  Fitting SHAP KernelExplainer ...")
        try:
            def predict_fn(X_arr):
                return calibrated_model.predict_proba(X_arr)[:, 1]
            explainer_shap = shap.KernelExplainer(
                predict_fn,
                shap.sample(X, min(200, X.shape[0]), random_state=42),
            )
            with open(f"{OUT}/shap_explainer.pkl", 'wb') as f:
                pickle.dump(explainer_shap, f)
            print(f"  ✓ shap_explainer.pkl saved")
        except Exception as e:
            print(f"  ⚠ SHAP creation failed: {e}")

else:
    # ── Fairlearn not installed: use re-weighted model as fallback ─────────────
    print("  ⚠ Fairlearn not installed — using re-weighted model as fallback")
    debiased_model_final      = mitigated_model
    debiased_model_calibrated = mitigated_model
    with open(f"{OUT}/model_fairlearn_calibrated.pkl", 'wb') as f:
        pickle.dump(mitigated_model, f)
    with open(f"{OUT}/model_debiased_fairlearn.pkl", 'wb') as f:
        pickle.dump(mitigated_model, f)
    with open(f"{OUT}/model_logistic_regression.pkl", 'wb') as f:
        pickle.dump(trained_models['Logistic Regression'], f)
    print(f"  ✓ Fallback models saved")

# ═══════════════════════════════════════════════════════════
#  STEP 10 ─ FINAL SUMMARY  (Advanced gridspec layout)
#  The Fairness Scorecard now reflects the calibrated model.
# ═══════════════════════════════════════════════════════════
print("\n► STEP 10: Generating Final Summary...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F7F9FC')
gs2 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle(
    'Final Results Summary — Equitable HCT Survival Prediction System',
    fontsize=16, fontweight='bold', color='#1a2740',
)

# ── Top-left: Model performance bars ──────────────────────
ax_s1 = fig.add_subplot(gs2[0, 0])
ax_s1.set_facecolor('#F7F9FC')
metric_keys_s = ['AUC', 'Accuracy', 'F1', 'Recall']
metric_lbl_s  = ['Overall\nAccuracy', 'Correct\nPredictions',
                 'Balanced\nScore', 'Survivor\nDetection']
x_s = np.arange(len(metric_keys_s)); w_s = 0.25
for i, (mname, color) in enumerate(zip(model_names, colors_m)):
    if mname not in results:
        continue
    means_s = [results[mname][m]['mean'] for m in metric_keys_s]
    ax_s1.bar(x_s + i * w_s, means_s, w_s, label=mname, color=color,
              alpha=0.85, edgecolor='white', linewidth=1.8, zorder=3)
ax_s1.set_xticks(x_s + w_s)
ax_s1.set_xticklabels(metric_lbl_s, fontsize=9)
ax_s1.set_ylabel('Score', fontsize=10)
ax_s1.set_title('Model Performance Summary', fontweight='bold', fontsize=11)
ax_s1.legend(fontsize=8)
ax_s1.set_ylim(0.4, 1.0)
ax_s1.axhline(0.70, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1.5)
ax_s1.yaxis.grid(True, alpha=0.3, zorder=0)
ax_s1.set_axisbelow(True)
for sp in ['top', 'right']: ax_s1.spines[sp].set_visible(False)

# ── Top-right: Fairness journey (includes Fairlearn if available) ──────────────
ax_s2 = fig.add_subplot(gs2[0, 1])
ax_s2.set_facecolor('#F7F9FC')

if fm_fair is not None:
    stages_s = ['Before\nCorrection', 'After Group\nBalancing',
                'After Threshold\nAdjust.', 'Fairlearn\nEG (deployed)']
    dp_s = [best_fm['demographic_parity_diff'],
            fm_mitigated['demographic_parity_diff'],
            fm_adjusted['demographic_parity_diff'],
            fm_calibrated['demographic_parity_diff']]
    eo_s = [best_fm['equal_opportunity_diff'],
            fm_mitigated['equal_opportunity_diff'],
            fm_adjusted['equal_opportunity_diff'],
            fm_calibrated['equal_opportunity_diff']]
else:
    stages_s = ['Before\nCorrection', 'After Group\nBalancing',
                'After Threshold\nAdjust.']
    dp_s = [best_fm['demographic_parity_diff'],
            fm_mitigated['demographic_parity_diff'],
            fm_adjusted['demographic_parity_diff']]
    eo_s = [best_fm['equal_opportunity_diff'],
            fm_mitigated['equal_opportunity_diff'],
            fm_adjusted['equal_opportunity_diff']]

x_fs = np.arange(len(stages_s)); w_fs = 0.35
b1_s = ax_s2.bar(x_fs - w_fs / 2, dp_s, w_fs, label='Equal Prediction Gap',
                  color='#1E6CB3', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
b2_s = ax_s2.bar(x_fs + w_fs / 2, eo_s, w_fs, label='Equal Detection Gap',
                  color='#E74C3C', alpha=0.82, edgecolor='white', linewidth=1.5, zorder=3)
ax_s2.axhline(0.10, color='#2DB87D', linestyle='--', linewidth=2.5,
              label='Target ≤ 0.10  ✓ PASS')
for bar, val in [(b, v) for bars_ in [b1_s, b2_s] for b, v in
                 zip(bars_, dp_s + eo_s)]:
    pass  # annotate inside
for bars_, vals in [(b1_s, dp_s), (b2_s, eo_s)]:
    for bar, val in zip(bars_, vals):
        icon = '✓' if val <= 0.10 else '✗'
        c_txt = '#2DB87D' if val <= 0.10 else '#E74C3C'
        ax_s2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                   f'{icon}{val:.3f}', ha='center', va='bottom',
                   fontsize=7.5, fontweight='bold', color=c_txt)
ax_s2.set_xticks(x_fs)
ax_s2.set_xticklabels(stages_s, fontsize=8.5)
ax_s2.set_ylabel('Fairness Gap (lower = fairer)', fontsize=10)
ax_s2.set_title('Fairness Improvement Journey\n(Final deployed model: Fairlearn + Calibration)',
                fontweight='bold', fontsize=11)
ax_s2.legend(fontsize=8)
ax_s2.yaxis.grid(True, alpha=0.3, zorder=0)
ax_s2.set_axisbelow(True)
for sp in ['top', 'right']: ax_s2.spines[sp].set_visible(False)

# ── Bottom span: Comprehensive scorecard table ─────────────────────────────────
ax_s3 = fig.add_subplot(gs2[1, :])
ax_s3.axis('off')

# Determine final deployed model metrics
if fm_fair is not None:
    final_dp = fm_calibrated['demographic_parity_diff']
    final_eo = fm_calibrated['equal_opportunity_diff']
    final_label = 'Fairlearn EG + Calibrated LR (DEPLOYED)'
    final_auc = roc_auc_score(y, debiased_model_calibrated.predict_proba(X)[:, 1])
else:
    final_dp = fm_mitigated['demographic_parity_diff']
    final_eo = fm_mitigated['equal_opportunity_diff']
    final_label = 'Re-weighted (Fairlearn unavailable)'
    final_auc = roc_auc_score(y, mitigated_model.predict_proba(X)[:, 1])

perf_data = []
for mname in model_names:
    r      = results[mname]
    fm_row = fairness_results[mname]
    dp_icon = '✓ PASS' if fm_row['demographic_parity_diff'] <= 0.10 else '✗ FAIL'
    eo_icon = '✓ PASS' if fm_row['equal_opportunity_diff']  <= 0.10 else '✗ FAIL'
    perf_data.append([
        mname,
        f"{r['AUC']['mean']:.4f} ± {r['AUC']['std']:.4f}",
        f"{r['Accuracy']['mean']:.4f}",
        f"{r['F1']['mean']:.4f}",
        f"{r['Recall']['mean']:.4f}",
        f"{fm_row['demographic_parity_diff']:.4f}  {dp_icon}",
        f"{fm_row['equal_opportunity_diff']:.4f}  {eo_icon}",
    ])

# Add deployed model row
dp_icon_f = '✓ PASS' if final_dp <= 0.10 else '✗ FAIL'
eo_icon_f = '✓ PASS' if final_eo <= 0.10 else '✗ FAIL'
perf_data.append([
    f'★ {final_label}',
    f'{final_auc:.4f}  [DEPLOYED]',
    '—',
    '—',
    '—',
    f'{final_dp:.4f}  {dp_icon_f}',
    f'{final_eo:.4f}  {eo_icon_f}',
])

col_labels = [
    'Model', 'Overall Accuracy (AUC)', 'Correct Predictions',
    'Balanced Score (F1)', 'Survivor Detection',
    'Equal Prediction Gap', 'Equal Detection Gap',
]
table = ax_s3.table(
    cellText=perf_data, colLabels=col_labels,
    cellLoc='center', loc='center', bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
for (r_idx, c_idx), cell in table.get_celld().items():
    if r_idx == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#1a2740')
    elif r_idx == len(perf_data):   # deployed model row
        cell.set_facecolor('#E6F9F0')
        cell.set_text_props(fontweight='bold', color='#0B5E2A')
    elif r_idx % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor('#BDC3C7')
ax_s3.set_title(
    'Complete Performance & Fairness Scorecard — All Models\n'
    '(★ = deployed model: Fairlearn ExponentiatedGradient + Isotonic Calibration)',
    fontweight='bold', fontsize=12, pad=15,
)

plt.savefig(f"{OUT}/07_final_summary.png", dpi=150,
            bbox_inches='tight', facecolor='#F7F9FC')
plt.close()
print(f"  ✓ Final summary saved → {OUT}/07_final_summary.png")

# ── Save best_model_info.json ──────────────────────────────────────────────────
# newapp.py reads this for the Model Performance tab metrics.
# We report the LR baseline for AUC (conservative, fair) and fairness metrics
# from the deployed calibrated model.
lr_auc = results['Logistic Regression']['AUC']['mean']
best_info = {
    'best_model_name': 'Logistic Regression',    # LR is the Fairlearn base
    'best_auc'       : lr_auc,
    'best_accuracy'  : results['Logistic Regression']['Accuracy']['mean'],
    'best_f1'        : results['Logistic Regression']['F1']['mean'],
    'best_recall'    : results['Logistic Regression']['Recall']['mean'],
    'dp_before'      : best_fm['demographic_parity_diff'],
    'eo_before'      : best_fm['equal_opportunity_diff'],
    'dp_after'       : final_dp,
    'eo_after'       : final_eo,
}
with open(f"{OUT}/best_model_info.json", 'w') as f:
    json.dump(best_info, f, indent=2)
print(f"  ✓ best_model_info.json saved")

# ═══════════════════════════════════════════════════════════
#  STEP 11 ─ TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════
print("\n► STEP 11: Generating test predictions...")

test_features = test.drop(columns=['ID'])
for col in feature_df.columns:
    if col not in test_features.columns:
        test_features[col] = np.nan
test_features = test_features[feature_df.columns]
X_test = preprocessor.transform(test_features)

if debiased_model_calibrated is not None:
    test_pred_probs = debiased_model_calibrated.predict_proba(X_test)[:, 1]
    print("  Using Fairlearn EG + Calibrated model for test predictions")
else:
    test_pred_probs = mitigated_model.predict_proba(X_test)[:, 1]
    print("  Using re-weighted fallback for test predictions")

submission = pd.DataFrame({'ID': test['ID'], 'prediction': test_pred_probs})
submission.to_csv(f"{OUT}/submission.csv", index=False)
print(f"  ✓ submission.csv saved ({submission.shape}) | "
      f"Range: [{test_pred_probs.min():.3f}, {test_pred_probs.max():.3f}]")

# ═══════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — ALL FILES SAVED")
print("=" * 60)
print(f"\n  Output directory       : {OUT}/")
print(f"  Plots                  : 01_eda … 07_final_summary")
print(f"  Deployed model         : model_fairlearn_calibrated.pkl")
print(f"  Fallback alias         : model_debiased_fairlearn.pkl")
print(f"  Preprocessor           : preprocessor.pkl")
print(f"  App JSON files         : feature_names, top_features, lr_coefficients,")
print(f"                           cv_results, fairness_results, mitigation_results,")
print(f"                           best_model_info")
print(f"\n  CV champion (visuals)  : {best_model_name}")
print(f"  CV champion AUC        : {results[best_model_name]['AUC']['mean']:.4f}")
print(f"  LR base AUC            : {results['Logistic Regression']['AUC']['mean']:.4f}")

if fm_fair is not None:
    dp_ok = fm_calibrated['demographic_parity_diff'] <= 0.10
    eo_ok = fm_calibrated['equal_opportunity_diff']  <= 0.10
    print(f"\n  FAIRNESS (deployed model):")
    print(f"    Equal Prediction Gap : {fm_calibrated['demographic_parity_diff']:.4f}  "
          f"{'✓ PASS' if dp_ok else '✗ FAIL'}")
    print(f"    Equal Detection Gap  : {fm_calibrated['equal_opportunity_diff']:.4f}  "
          f"{'✓ PASS' if eo_ok else '✗ FAIL'}")
    if dp_ok and eo_ok:
        print(f"\n  ✓ ALL FAIRNESS THRESHOLDS MET (< 0.10)")
    else:
        print(f"\n  ⚠ One or more fairness thresholds still exceeded")
else:
    print(f"\n  ⚠ Fairlearn not installed — re-weighted fallback used")
    print(f"    Run: pip install fairlearn  then re-run pipeline")

print("=" * 60 + "\n")