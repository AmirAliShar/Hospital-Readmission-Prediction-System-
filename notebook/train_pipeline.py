"""
Hospital Readmission Prediction — Complete ML Training Pipeline
================================================================
UCI Diabetes 130-US Hospitals Dataset
Generates: trained model, metrics, EDA data, charts for backend/frontend
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "diabetic_data.csv"
ARTIFACTS_DIR = BASE_DIR / "backend" / "artifacts"
REPORTS_DIR = BASE_DIR / "backend" / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
SCREENSHOTS_DIR = BASE_DIR / "screenshots" / "model-evaluation"

for d in [ARTIFACTS_DIR, REPORTS_DIR, CHARTS_DIR, SCREENSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  Hospital Readmission Prediction — Training Pipeline")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/12] Loading dataset...")
df_raw = pd.read_csv(DATA_PATH, na_values='?')
print(f"  Shape: {df_raw.shape}")
print(f"  Columns: {len(df_raw.columns)}")
print(f"  Memory: {df_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/12] Data quality assessment...")
missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
missing_report = missing_pct[missing_pct > 0]
print(f"  Columns with missing values: {len(missing_report)}")
for col, pct in missing_report.items():
    print(f"    {col}: {pct:.1f}%")

# Save data quality report
data_quality = {
    "shape": list(df_raw.shape),
    "n_rows": int(df_raw.shape[0]),
    "n_columns": int(df_raw.shape[1]),
    "memory_mb": round(df_raw.memory_usage(deep=True).sum() / 1e6, 1),
    "missing_values": {col: {"count": int(df_raw[col].isnull().sum()), "percent": round(pct, 1)}
                       for col, pct in missing_report.items()},
    "duplicates": int(df_raw.duplicated().sum()),
    "column_types": {str(k): int(v) for k, v in df_raw.dtypes.value_counts().items()},
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TARGET ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/12] Engineering binary target...")
target_dist_raw = df_raw['readmitted'].value_counts()
print(f"  Original distribution:\n{target_dist_raw.to_string()}")

df = df_raw.copy()
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
target_dist = df['readmitted_binary'].value_counts()
print(f"\n  Binary target (1 = readmitted <30 days):")
print(f"    0 (Not <30): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
print(f"    1 (<30 days): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")

target_info = {
    "original_distribution": {str(k): int(v) for k, v in target_dist_raw.items()},
    "binary_distribution": {str(k): int(v) for k, v in target_dist.items()},
    "positive_rate": round(target_dist[1] / len(df) * 100, 2),
    "imbalance_ratio": round(target_dist[0] / target_dist[1], 2),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/12] Cleaning data...")

# Remove deceased patients (discharge_disposition_id 11, 13, 14, 19, 20, 21)
deceased_ids = [11, 13, 14, 19, 20, 21]
n_before = len(df)
df = df[~df['discharge_disposition_id'].isin(deceased_ids)]
print(f"  Removed {n_before - len(df)} deceased/hospice patients")

# Keep first encounter per patient to avoid patient-level leakage
n_before = len(df)
df = df.sort_values('encounter_id').groupby('patient_nbr').first().reset_index()
print(f"  Kept first encounter per patient: {n_before} -> {len(df)} rows")

# Drop columns with excessive missing values or low information
drop_cols = [
    'encounter_id',     # ID - not a feature
    'patient_nbr',      # ID - not a feature
    'weight',           # 97% missing
    'payer_code',       # 40% missing, not clinically informative for prediction
    'medical_specialty', # 49% missing
    'readmitted',       # Original multi-class target (replaced by binary)
    # Near-zero variance medication columns
    'acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'glipizide-metformin',
]
existing_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop)
print(f"  Dropped {len(existing_drop)} columns (IDs, high-missing, near-zero variance)")

# Remove 'Unknown/Invalid' gender
n_before = len(df)
df = df[df['gender'].isin(['Male', 'Female'])]
print(f"  Removed {n_before - len(df)} rows with unknown gender")

print(f"  Final cleaned shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/12] Engineering features...")

# Age: map to numeric midpoints
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_map)

# Diagnosis category from diag_1
def categorize_diag(value):
    try:
        v = float(value)
        if 390 <= v <= 459 or v == 785:
            return 'Circulatory'
        elif 460 <= v <= 519 or v == 786:
            return 'Respiratory'
        elif 520 <= v <= 579 or v == 787:
            return 'Digestive'
        elif 250 <= v < 251:
            return 'Diabetes'
        elif 800 <= v <= 999:
            return 'Injury'
        elif 710 <= v <= 739:
            return 'Musculoskeletal'
        elif 580 <= v <= 629 or v == 788:
            return 'Genitourinary'
        elif 140 <= v <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except (ValueError, TypeError):
        return 'Other'

df['diag_1_category'] = df['diag_1'].apply(categorize_diag)

# Total past admissions
df['total_past_admissions'] = (
    df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
)

# Polypharmacy indicator
poly_threshold = df['num_medications'].quantile(0.75)
df['polypharmacy'] = (df['num_medications'] > poly_threshold).astype(int)

# Admission risk level
def admission_risk(x):
    if x == 0:
        return 'None'
    elif 1 <= x <= 3:
        return 'Low'
    elif 4 <= x <= 10:
        return 'Medium'
    else:
        return 'High'

df['admission_risk_level'] = df['total_past_admissions'].apply(admission_risk)

# Number of medication changes
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'insulin',
            'glyburide-metformin']
med_cols = [c for c in med_cols if c in df.columns]
df['n_med_changes'] = df[med_cols].apply(
    lambda row: sum(1 for v in row if v in ['Up', 'Down']), axis=1
)

# Has insulin change
if 'insulin' in df.columns:
    df['insulin_change'] = df['insulin'].apply(lambda x: 1 if x in ['Up', 'Down'] else 0)

# Binary encode simple columns
df['change_binary'] = (df['change'] == 'Ch').astype(int)
df['diabetesMed_binary'] = (df['diabetesMed'] == 'Yes').astype(int)
df['A1C_abnormal'] = df['A1Cresult'].apply(lambda x: 1 if x in ['>7', '>8'] else 0) if 'A1Cresult' in df.columns else 0
df['max_glu_abnormal'] = df['max_glu_serum'].apply(lambda x: 1 if x in ['>200', '>300'] else 0) if 'max_glu_serum' in df.columns else 0

print(f"  Created 8 engineered features")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EDA CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/12] Generating EDA charts...")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Chart 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
target_counts = df['readmitted_binary'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Not Readmitted (0)', 'Readmitted <30d (1)'], target_counts.values, color=colors)
axes[0].set_title('Readmission Target Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
axes[1].pie(target_counts.values, labels=['Not <30d', 'Readmitted <30d'],
            colors=colors, autopct='%1.1f%%', startangle=90,
            explode=(0, 0.05), shadow=True)
axes[1].set_title('Class Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'target_distribution.png', bbox_inches='tight')
plt.close()

# Chart 2: Age distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
age_order = sorted(df['age'].dropna().unique(), key=lambda x: age_map.get(x, 0))
age_counts = df['age'].value_counts().reindex(age_order)
axes[0].bar(range(len(age_counts)), age_counts.values, color='#3498db')
axes[0].set_xticks(range(len(age_counts)))
axes[0].set_xticklabels(age_counts.index, rotation=45)
axes[0].set_title('Age Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
readmit_by_age = df.groupby('age')['readmitted_binary'].mean().reindex(age_order) * 100
axes[1].bar(range(len(readmit_by_age)), readmit_by_age.values, color='#e74c3c')
axes[1].set_xticks(range(len(readmit_by_age)))
axes[1].set_xticklabels(readmit_by_age.index, rotation=45)
axes[1].set_title('Readmission Rate by Age Group', fontweight='bold')
axes[1].set_ylabel('Readmission Rate (%)')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'age_analysis.png', bbox_inches='tight')
plt.close()

# Chart 3: Gender and Race distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gender_readmit = df.groupby('gender')['readmitted_binary'].agg(['mean', 'count']).reset_index()
bars = axes[0].bar(gender_readmit['gender'], gender_readmit['mean'] * 100, color=['#e91e63', '#2196f3'])
axes[0].set_title('Readmission Rate by Gender', fontweight='bold')
axes[0].set_ylabel('Readmission Rate (%)')
for bar, count in zip(bars, gender_readmit['count']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'n={count:,}', ha='center', fontsize=9)
race_readmit = df.groupby('race')['readmitted_binary'].agg(['mean', 'count']).reset_index()
race_readmit = race_readmit.sort_values('mean', ascending=False)
bars = axes[1].barh(race_readmit['race'], race_readmit['mean'] * 100, color='#9b59b6')
axes[1].set_title('Readmission Rate by Race', fontweight='bold')
axes[1].set_xlabel('Readmission Rate (%)')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'gender_race_analysis.png', bbox_inches='tight')
plt.close()

# Chart 4: Diagnosis category
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
diag_counts = df['diag_1_category'].value_counts()
axes[0].barh(diag_counts.index, diag_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(diag_counts))))
axes[0].set_title('Primary Diagnosis Distribution', fontweight='bold')
axes[0].set_xlabel('Count')
diag_readmit = df.groupby('diag_1_category')['readmitted_binary'].mean().sort_values(ascending=False) * 100
axes[1].barh(diag_readmit.index, diag_readmit.values, color='#e74c3c')
axes[1].set_title('Readmission Rate by Diagnosis', fontweight='bold')
axes[1].set_xlabel('Readmission Rate (%)')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'diagnosis_analysis.png', bbox_inches='tight')
plt.close()

# Chart 5: Hospital stay & medications
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(df['time_in_hospital'], bins=14, color='#3498db', edgecolor='white', alpha=0.8)
axes[0,0].set_title('Time in Hospital (days)', fontweight='bold')
axes[0,0].set_xlabel('Days')
axes[0,1].hist(df['num_medications'], bins=30, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[0,1].set_title('Number of Medications', fontweight='bold')
axes[0,1].set_xlabel('Medications')
axes[1,0].hist(df['num_lab_procedures'], bins=30, color='#e67e22', edgecolor='white', alpha=0.8)
axes[1,0].set_title('Number of Lab Procedures', fontweight='bold')
axes[1,0].set_xlabel('Procedures')
axes[1,1].hist(df['number_diagnoses'], bins=16, color='#9b59b6', edgecolor='white', alpha=0.8)
axes[1,1].set_title('Number of Diagnoses', fontweight='bold')
axes[1,1].set_xlabel('Diagnoses')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'distributions.png', bbox_inches='tight')
plt.close()

# Chart 6: Emergency & outpatient visits
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
visit_data = df.groupby('readmitted_binary')[['number_emergency', 'number_outpatient', 'number_inpatient']].mean()
visit_data.T.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Mean Visit Counts by Readmission', fontweight='bold')
axes[0].set_ylabel('Mean Count')
axes[0].legend(['Not Readmitted', 'Readmitted <30d'])
axes[0].tick_params(axis='x', rotation=45)
stay_readmit = df.groupby('time_in_hospital')['readmitted_binary'].mean() * 100
axes[1].plot(stay_readmit.index, stay_readmit.values, 'o-', color='#e74c3c', markersize=5)
axes[1].set_title('Readmission Rate by Hospital Stay Duration', fontweight='bold')
axes[1].set_xlabel('Days in Hospital')
axes[1].set_ylabel('Readmission Rate (%)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'visits_stay_analysis.png', bbox_inches='tight')
plt.close()

# Chart 7: Missing values
fig, ax = plt.subplots(figsize=(10, 5))
missing_df = pd.DataFrame({
    'Column': list(data_quality['missing_values'].keys()),
    'Percent': [v['percent'] for v in data_quality['missing_values'].values()]
})
missing_df = missing_df.sort_values('Percent', ascending=True)
ax.barh(missing_df['Column'], missing_df['Percent'], color='#e74c3c')
ax.set_title('Missing Values in Original Dataset', fontweight='bold')
ax.set_xlabel('Missing (%)')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'missing_values.png', bbox_inches='tight')
plt.close()

print(f"  Saved 7 EDA chart files to {CHARTS_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PREPARE MODELING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/12] Preparing features for modeling...")

# Select final features
feature_cols_numeric = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'age_numeric',
    'total_past_admissions', 'n_med_changes',
    'polypharmacy', 'diabetesMed_binary', 'change_binary',
    'insulin_change', 'A1C_abnormal', 'max_glu_abnormal',
    'admission_type_id',
]

feature_cols_categorical = [
    'gender', 'race', 'diag_1_category', 'admission_risk_level',
]

# Remove any features not present
feature_cols_numeric = [c for c in feature_cols_numeric if c in df.columns]
feature_cols_categorical = [c for c in feature_cols_categorical if c in df.columns]
all_feature_cols = feature_cols_numeric + feature_cols_categorical

target_col = 'readmitted_binary'

print(f"  Numeric features: {len(feature_cols_numeric)}")
print(f"  Categorical features: {len(feature_cols_categorical)}")
print(f"  Total features: {len(all_feature_cols)}")

# Fill remaining NaN in numeric cols
for col in feature_cols_numeric:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Fill remaining NaN in categorical cols
for col in feature_cols_categorical:
    if df[col].isnull().any():
        df[col] = df[col].fillna('Unknown')

X = df[all_feature_cols].copy()
y = df[target_col].copy()

print(f"  X shape: {X.shape}, y shape: {y.shape}")
print(f"  Positive rate: {y.mean()*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[8/12] Splitting data...")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"  Train: {X_train.shape[0]} ({y_train.mean()*100:.1f}% positive)")
print(f"  Val:   {X_val.shape[0]} ({y_val.mean()*100:.1f}% positive)")
print(f"  Test:  {X_test.shape[0]} ({y_test.mean()*100:.1f}% positive)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[9/12] Building preprocessing pipeline...")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# For tree-based models: ordinal encoding is fine
cat_categories = {}
for col in feature_cols_categorical:
    cat_categories[col] = sorted(X_train[col].unique().tolist())

ordinal_preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), feature_cols_numeric),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                        categories=[cat_categories[c] for c in feature_cols_categorical])),
        ]), feature_cols_categorical),
    ],
    remainder='drop'
)

# For linear models: one-hot encoding + scaling
linear_preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), feature_cols_numeric),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), feature_cols_categorical),
    ],
    remainder='drop'
)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[10/12] Training models...")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, log_loss, brier_score_loss,
    roc_curve, precision_recall_curve
)

# Calculate scale_pos_weight for imbalanced classes
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_weight = n_neg / n_pos
print(f"  Class weight ratio: {scale_weight:.2f}")

# Define models
models = {}

# 1. Logistic Regression
models['Logistic Regression'] = {
    'pipeline': Pipeline([
        ('preprocessor', linear_preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight='balanced', C=0.1, solver='lbfgs'
        ))
    ]),
    'type': 'linear'
}

# 2. Random Forest
models['Random Forest'] = {
    'pipeline': Pipeline([
        ('preprocessor', ordinal_preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ]),
    'type': 'tree'
}

# 3. XGBoost
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = {
        'pipeline': Pipeline([
            ('preprocessor', ordinal_preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                random_state=42, n_jobs=-1,
                eval_metric='logloss', verbosity=0
            ))
        ]),
        'type': 'tree'
    }
except ImportError:
    print("  WARNING: XGBoost not available")

# 4. LightGBM
try:
    from lightgbm import LGBMClassifier
    models['LightGBM'] = {
        'pipeline': Pipeline([
            ('preprocessor', ordinal_preprocessor),
            ('classifier', LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                random_state=42, n_jobs=-1, verbose=-1
            ))
        ]),
        'type': 'tree'
    }
except ImportError:
    print("  WARNING: LightGBM not available")

# 5. HistGradientBoosting
models['HistGradientBoosting'] = {
    'pipeline': Pipeline([
        ('preprocessor', ordinal_preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            random_state=42, class_weight='balanced'
        ))
    ]),
    'type': 'tree'
}

# Train and evaluate all models
results = {}

def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    print(f"\n  Training {name}...")
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]

    metrics = {
        'accuracy': round(float(accuracy_score(y_te, y_pred)), 4),
        'precision': round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_te, y_proba)), 4),
        'pr_auc': round(float(average_precision_score(y_te, y_proba)), 4),
        'log_loss': round(float(log_loss(y_te, y_proba)), 4),
        'brier_score': round(float(brier_score_loss(y_te, y_proba)), 4),
    }

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    metrics['specificity'] = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0
    metrics['confusion_matrix'] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    # Classification report
    metrics['classification_report'] = classification_report(y_te, y_pred, output_dict=True, zero_division=0)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    metrics['roc_curve'] = {
        'fpr': [round(float(x), 4) for x in fpr[::max(1, len(fpr)//100)]],
        'tpr': [round(float(x), 4) for x in tpr[::max(1, len(tpr)//100)]],
    }

    # PR curve data
    prec, rec, _ = precision_recall_curve(y_te, y_proba)
    metrics['pr_curve'] = {
        'precision': [round(float(x), 4) for x in prec[::max(1, len(prec)//100)]],
        'recall': [round(float(x), 4) for x in rec[::max(1, len(rec)//100)]],
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    return metrics

for name, model_info in models.items():
    try:
        metrics = evaluate_model(name, model_info['pipeline'], X_train, y_train, X_val, y_val)
        results[name] = {
            'metrics': metrics,
            'pipeline': model_info['pipeline'],
            'type': model_info['type'],
        }
    except Exception as e:
        print(f"  ERROR training {name}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. MODEL SELECTION & FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[11/12] Selecting best model and final evaluation...")

# Select based on ROC-AUC (suitable for imbalanced healthcare data)
best_name = max(results, key=lambda k: results[k]['metrics']['roc_auc'])
print(f"\n  Best model by ROC-AUC: {best_name}")
print(f"  Validation ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f}")

# Retrain best model on full train+val for final model
best_pipeline = results[best_name]['pipeline']
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
best_pipeline.fit(X_train_full, y_train_full)

# Final evaluation on test set
y_pred_final = best_pipeline.predict(X_test)
y_proba_final = best_pipeline.predict_proba(X_test)[:, 1]

final_test_metrics = {
    'accuracy': round(float(accuracy_score(y_test, y_pred_final)), 4),
    'precision': round(float(precision_score(y_test, y_pred_final, zero_division=0)), 4),
    'recall': round(float(recall_score(y_test, y_pred_final, zero_division=0)), 4),
    'f1_score': round(float(f1_score(y_test, y_pred_final, zero_division=0)), 4),
    'roc_auc': round(float(roc_auc_score(y_test, y_proba_final)), 4),
    'pr_auc': round(float(average_precision_score(y_test, y_proba_final)), 4),
    'log_loss': round(float(log_loss(y_test, y_proba_final)), 4),
    'brier_score': round(float(brier_score_loss(y_test, y_proba_final)), 4),
}
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
final_test_metrics['specificity'] = round(float(tn / (tn + fp)), 4)
final_test_metrics['confusion_matrix'] = [[int(tn), int(fp)], [int(fn), int(tp)]]

fpr, tpr, _ = roc_curve(y_test, y_proba_final)
final_test_metrics['roc_curve'] = {
    'fpr': [round(float(x), 4) for x in fpr[::max(1, len(fpr)//100)]],
    'tpr': [round(float(x), 4) for x in tpr[::max(1, len(tpr)//100)]],
}
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba_final)
final_test_metrics['pr_curve'] = {
    'precision': [round(float(x), 4) for x in prec_arr[::max(1, len(prec_arr)//100)]],
    'recall': [round(float(x), 4) for x in rec_arr[::max(1, len(rec_arr)//100)]],
}

print(f"\n  Final Test Metrics ({best_name}):")
for k, v in final_test_metrics.items():
    if k not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
        print(f"    {k}: {v}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. GENERATE EVALUATION CHARTS & SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[12/12] Generating evaluation charts and saving artifacts...")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Readmitted', 'Readmitted <30d'],
            yticklabels=['Not Readmitted', 'Readmitted <30d'])
ax.set_title(f'Confusion Matrix - {best_name}', fontweight='bold', fontsize=14)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'confusion_matrix.png', bbox_inches='tight')
plt.savefig(SCREENSHOTS_DIR / 'confusion_matrix.png', bbox_inches='tight')
plt.close()

# ROC Curve - all models
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = plt.cm.Set1(np.linspace(0, 1, len(results)))
for (name, res), color in zip(results.items(), colors_roc):
    fpr_v = res['metrics']['roc_curve']['fpr']
    tpr_v = res['metrics']['roc_curve']['tpr']
    ax.plot(fpr_v, tpr_v, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'roc_curves.png', bbox_inches='tight')
plt.savefig(SCREENSHOTS_DIR / 'roc_curves.png', bbox_inches='tight')
plt.close()

# PR Curve - all models
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), colors_roc):
    prec_v = res['metrics']['pr_curve']['precision']
    rec_v = res['metrics']['pr_curve']['recall']
    ax.plot(rec_v, prec_v, label=f"{name} (AP={res['metrics']['pr_auc']:.3f})", color=color, linewidth=2)
baseline = y_test.mean()
ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves - Model Comparison', fontweight='bold', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'pr_curves.png', bbox_inches='tight')
plt.savefig(SCREENSHOTS_DIR / 'pr_curves.png', bbox_inches='tight')
plt.close()

# Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
comparison_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'recall', 'precision']
model_names = list(results.keys())
x = np.arange(len(comparison_metrics))
width = 0.8 / len(model_names)
for i, name in enumerate(model_names):
    values = [results[name]['metrics'][m] for m in comparison_metrics]
    axes[0].bar(x + i * width, values, width, label=name, alpha=0.8)
axes[0].set_xticks(x + width * (len(model_names) - 1) / 2)
axes[0].set_xticklabels([m.replace('_', ' ').title() for m in comparison_metrics], rotation=30)
axes[0].set_title('Model Comparison (Validation Set)', fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')
loss_metrics = ['log_loss', 'brier_score']
x2 = np.arange(len(loss_metrics))
for i, name in enumerate(model_names):
    values = [results[name]['metrics'][m] for m in loss_metrics]
    axes[1].bar(x2 + i * width, values, width, label=name, alpha=0.8)
axes[1].set_xticks(x2 + width * (len(model_names) - 1) / 2)
axes[1].set_xticklabels([m.replace('_', ' ').title() for m in loss_metrics])
axes[1].set_title('Loss Metrics (lower = better)', fontweight='bold')
axes[1].set_ylabel('Score')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'model_comparison.png', bbox_inches='tight')
plt.savefig(SCREENSHOTS_DIR / 'model_comparison.png', bbox_inches='tight')
plt.close()

# Feature Importance
feature_importance_data = {}
try:
    classifier = best_pipeline.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        preprocessor = best_pipeline.named_steps['preprocessor']
        feature_names_out = feature_cols_numeric + feature_cols_categorical
        importances = classifier.feature_importances_
        if len(importances) == len(feature_names_out):
            feat_imp = pd.Series(importances, index=feature_names_out).sort_values(ascending=True)
        else:
            feat_imp = pd.Series(importances).sort_values(ascending=True)
            feat_imp.index = [f'Feature_{i}' for i in range(len(feat_imp))]

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(feat_imp))
        feat_imp.tail(top_n).plot(kind='barh', ax=ax, color='#3498db')
        ax.set_title(f'Top {top_n} Feature Importances - {best_name}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / 'feature_importance.png', bbox_inches='tight')
        plt.savefig(SCREENSHOTS_DIR / 'feature_importance.png', bbox_inches='tight')
        plt.close()
        feature_importance_data = {name: round(float(val), 6) for name, val in feat_imp.items()}
except Exception as e:
    print(f"  Feature importance error: {e}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
corr_data = df[feature_cols_numeric + [target_col]].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', fmt='.2f',
            linewidths=0.5, ax=ax, center=0, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'correlation_heatmap.png', bbox_inches='tight')
plt.close()

print(f"  Saved evaluation charts")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Saving artifacts...")
print("=" * 70)

import joblib

# Save the best pipeline
joblib.dump(best_pipeline, ARTIFACTS_DIR / 'model_pipeline.joblib')
print(f"  Model saved: {(ARTIFACTS_DIR / 'model_pipeline.joblib').stat().st_size / 1e6:.1f} MB")

# Save model metadata
model_metadata = {
    "model_name": best_name,
    "model_version": "2.0",
    "dataset": "UCI Diabetes 130-US Hospitals",
    "dataset_rows": int(len(df)),
    "dataset_columns": int(df.shape[1]),
    "target": "readmitted_binary (1 = readmitted within 30 days)",
    "positive_rate_percent": round(float(y.mean() * 100), 2),
    "feature_names_numeric": feature_cols_numeric,
    "feature_names_categorical": feature_cols_categorical,
    "feature_count": len(all_feature_cols),
    "train_size": int(len(X_train_full)),
    "test_size": int(len(X_test)),
    "selection_metric": "ROC-AUC",
    "selection_rationale": f"{best_name} selected for highest validation ROC-AUC among {len(results)} models evaluated. ROC-AUC is appropriate for imbalanced healthcare prediction tasks as it measures discrimination ability across all thresholds.",
    "decision_threshold": 0.5,
    "training_date": datetime.now().isoformat(),
    "random_seed": 42,
    "packages": {
        "scikit-learn": __import__('sklearn').__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
}
try:
    import xgboost
    model_metadata["packages"]["xgboost"] = xgboost.__version__
except:
    pass
try:
    import lightgbm
    model_metadata["packages"]["lightgbm"] = lightgbm.__version__
except:
    pass

with open(ARTIFACTS_DIR / 'model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

# Save validation metrics for all models
validation_results = {}
for name, res in results.items():
    m = res['metrics'].copy()
    summary = {k: v for k, v in m.items() if k not in ['roc_curve', 'pr_curve', 'classification_report']}
    validation_results[name] = summary

with open(REPORTS_DIR / 'validation_metrics.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

# Save test metrics
with open(REPORTS_DIR / 'test_metrics.json', 'w') as f:
    json.dump(final_test_metrics, f, indent=2, default=str)

# Save full results with curves for frontend
full_results = {}
for name, res in results.items():
    full_results[name] = res['metrics']
with open(REPORTS_DIR / 'model_comparison.json', 'w') as f:
    json.dump(full_results, f, indent=2)

# Save feature importance
with open(REPORTS_DIR / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance_data, f, indent=2)

# Save EDA summary for the frontend
eda_summary = {
    "data_quality": data_quality,
    "target_info": target_info,
    "age_distribution": {str(k): int(v) for k, v in df['age'].value_counts().items()},
    "age_readmission_rate": {str(k): round(float(v)*100, 2) for k, v in df.groupby('age')['readmitted_binary'].mean().items()},
    "gender_distribution": {str(k): int(v) for k, v in df['gender'].value_counts().items()},
    "gender_readmission_rate": {str(k): round(float(v)*100, 2) for k, v in df.groupby('gender')['readmitted_binary'].mean().items()},
    "race_distribution": {str(k): int(v) for k, v in df['race'].value_counts().items() if pd.notna(k)},
    "race_readmission_rate": {str(k): round(float(v)*100, 2) for k, v in df.groupby('race')['readmitted_binary'].mean().items() if pd.notna(k)},
    "diagnosis_distribution": {str(k): int(v) for k, v in df['diag_1_category'].value_counts().items()},
    "diagnosis_readmission_rate": {str(k): round(float(v)*100, 2) for k, v in df.groupby('diag_1_category')['readmitted_binary'].mean().items()},
    "time_in_hospital_stats": {
        "mean": round(float(df['time_in_hospital'].mean()), 2),
        "median": round(float(df['time_in_hospital'].median()), 2),
        "std": round(float(df['time_in_hospital'].std()), 2),
        "distribution": {str(k): int(v) for k, v in df['time_in_hospital'].value_counts().sort_index().items()},
    },
    "num_medications_stats": {
        "mean": round(float(df['num_medications'].mean()), 2),
        "median": round(float(df['num_medications'].median()), 2),
        "std": round(float(df['num_medications'].std()), 2),
    },
    "num_lab_procedures_stats": {
        "mean": round(float(df['num_lab_procedures'].mean()), 2),
        "median": round(float(df['num_lab_procedures'].median()), 2),
    },
    "emergency_visits_stats": {
        "mean": round(float(df['number_emergency'].mean()), 2),
        "zero_pct": round(float((df['number_emergency'] == 0).mean() * 100), 1),
    },
    "outpatient_visits_stats": {
        "mean": round(float(df['number_outpatient'].mean()), 2),
        "zero_pct": round(float((df['number_outpatient'] == 0).mean() * 100), 1),
    },
    "feature_correlations": {col: round(float(corr_data.loc[col, target_col]), 4)
                             for col in feature_cols_numeric if col in corr_data.index},
    "charts": [
        "target_distribution.png", "age_analysis.png", "gender_race_analysis.png",
        "diagnosis_analysis.png", "distributions.png", "visits_stay_analysis.png",
        "missing_values.png", "correlation_heatmap.png",
    ],
    "evaluation_charts": [
        "confusion_matrix.png", "roc_curves.png", "pr_curves.png",
        "model_comparison.png", "feature_importance.png",
    ],
}
with open(REPORTS_DIR / 'eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2)

# Save allowed values for API validation
allowed_values = {
    "gender": sorted(df['gender'].unique().tolist()),
    "race": sorted([str(x) for x in df['race'].unique() if pd.notna(x)]),
    "diag_1_category": sorted(df['diag_1_category'].unique().tolist()),
    "admission_risk_level": ['None', 'Low', 'Medium', 'High'],
    "admission_type_id": sorted([int(x) for x in df['admission_type_id'].unique()]),
    "numeric_ranges": {
        "time_in_hospital": {"min": 1, "max": 14},
        "num_lab_procedures": {"min": 1, "max": 132},
        "num_procedures": {"min": 0, "max": 6},
        "num_medications": {"min": 1, "max": 81},
        "number_outpatient": {"min": 0, "max": 42},
        "number_emergency": {"min": 0, "max": 76},
        "number_inpatient": {"min": 0, "max": 21},
        "number_diagnoses": {"min": 1, "max": 16},
        "age_numeric": {"min": 5, "max": 95},
    }
}
with open(ARTIFACTS_DIR / 'allowed_values.json', 'w') as f:
    json.dump(allowed_values, f, indent=2)

print("\n" + "=" * 70)
print("  Pipeline complete! All artifacts saved.")
print("=" * 70)
print(f"\n  Model: {best_name}")
print(f"  Test ROC-AUC: {final_test_metrics['roc_auc']}")
print(f"  Test PR-AUC: {final_test_metrics['pr_auc']}")
print(f"  Test F1: {final_test_metrics['f1_score']}")
print(f"  Artifacts: {ARTIFACTS_DIR}")
print(f"  Reports: {REPORTS_DIR}")
print(f"  Charts: {CHARTS_DIR}")
