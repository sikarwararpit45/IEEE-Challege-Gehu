import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import early_stopping, log_evaluation


# -----------------------------
# Load Data
# -----------------------------
train = pd.read_csv("C:\\Users\Arpit\Downloads\\alrIEEEna_26_dataset\\ML Challenge Dataset\\TRAIN.csv")
test = pd.read_csv("C:\\Users\Arpit\Downloads\\alrIEEEna_26_dataset\\ML Challenge Dataset\\TEST.csv")

X = train.drop("Class", axis=1)
y = train["Class"]

X_test = test.drop("ID", axis=1)

base_cols = X.columns


# -----------------------------
# Feature Engineering
# -----------------------------
for df in [X, X_test]:

    df["log_F38"] = np.log1p(np.abs(df["F38"]))

    df["mean"] = df[base_cols].mean(axis=1)
    df["std"] = df[base_cols].std(axis=1)
    df["median"] = df[base_cols].median(axis=1)

    df["max"] = df[base_cols].max(axis=1)
    df["min"] = df[base_cols].min(axis=1)
    df["range"] = df["max"] - df["min"]

    df["sum"] = df[base_cols].sum(axis=1)

    df["skew"] = df[base_cols].skew(axis=1)
    df["kurtosis"] = df[base_cols].kurtosis(axis=1)

    df["q25"] = df[base_cols].quantile(0.25, axis=1)
    df["q75"] = df[base_cols].quantile(0.75, axis=1)

    df["cv"] = df["std"] / (df["mean"] + 1e-6)


# -----------------------------
# Manual Interaction Features
# -----------------------------
X["F01_F02"] = X["F01"] * X["F02"]
X["F10_F11"] = X["F10"] * X["F11"]
X["F20_F21"] = X["F20"] * X["F21"]

X_test["F01_F02"] = X_test["F01"] * X_test["F02"]
X_test["F10_F11"] = X_test["F10"] * X_test["F11"]
X_test["F20_F21"] = X_test["F20"] * X_test["F21"]


# -----------------------------
# Kaggle Trick: Auto Feature Interactions
# -----------------------------
print("Generating automatic interaction features...")

temp_model = LGBMClassifier(n_estimators=300, n_jobs=-1)
temp_model.fit(X, y)

importances = pd.Series(
    temp_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_features = importances.head(10).index

for i in range(len(top_features)):
    for j in range(i + 1, len(top_features)):

        f1 = top_features[i]
        f2 = top_features[j]

        new_col = f"{f1}_x_{f2}"

        X[new_col] = X[f1] * X[f2]
        X_test[new_col] = X_test[f1] * X_test[f2]

print("Interaction features created:", len(top_features)*(len(top_features)-1)//2)


# -----------------------------
# Cross Validation
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

fold_scores = []


for fold, (tr, va) in enumerate(skf.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    X_train, X_val = X.iloc[tr], X.iloc[va]
    y_train, y_val = y.iloc[tr], y.iloc[va]


    # -----------------------------
    # LightGBM (GPU)
    # -----------------------------
    lgb = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=3,
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )

    lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[
            early_stopping(300),
            log_evaluation(0)
        ]
    )

    oof_lgb[va] = lgb.predict_proba(X_val)[:,1]
    pred_lgb += lgb.predict_proba(X_test)[:,1] / 5


    # -----------------------------
    # XGBoost (GPU)
    # -----------------------------
    xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        max_bin=256
    )

    xgb.fit(X_train, y_train)

    oof_xgb[va] = xgb.predict_proba(X_val)[:,1]
    pred_xgb += xgb.predict_proba(X_test)[:,1] / 5


    # -----------------------------
    # CatBoost (GPU)
    # -----------------------------
    cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=6,
        task_type="GPU",
        devices="0",
        verbose=0
    )

    cat.fit(X_train, y_train)

    oof_cat[va] = cat.predict_proba(X_val)[:,1]
    pred_cat += cat.predict_proba(X_test)[:,1] / 5


    # -----------------------------
    # Ensemble Validation Score
    # -----------------------------
    val_pred = (
        0.45 * oof_lgb[va] +
        0.35 * oof_xgb[va] +
        0.20 * oof_cat[va]
    )

    val_class = (val_pred > 0.5).astype(int)

    acc = accuracy_score(y_val, val_class)

    print("Fold Validation Accuracy:", acc)

    fold_scores.append(acc)


# -----------------------------
# OOF Evaluation
# -----------------------------
oof = 0.45 * oof_lgb + 0.35 * oof_xgb + 0.20 * oof_cat

oof_pred = (oof > 0.5).astype(int)

oof_acc = accuracy_score(y, oof_pred)


print("\n=============================")
print("Fold Scores:", fold_scores)
print("Mean CV Score:", np.mean(fold_scores))
print("OOF Accuracy:", oof_acc)
print("=============================")


# -----------------------------
# Final Ensemble Prediction
# -----------------------------
pred = (
    0.45 * pred_lgb +
    0.35 * pred_xgb +
    0.20 * pred_cat
)

submission = pd.DataFrame({
    "ID": test["ID"],
    "Class": (pred > 0.5).astype(int)
})

submission.to_csv("FINAL.csv", index=False)

print("Submission saved.")