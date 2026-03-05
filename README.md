# Machine Learning Ensemble Model for Binary Classification

This project implements a high-performance machine learning pipeline for binary classification using feature engineering, gradient boosting algorithms, and ensemble learning.

The system predicts the target class (`Class`) from structured tabular data by combining multiple machine learning models and engineered features.

**Final performance**
- Mean Cross-Validation Accuracy: ~98.47%
- Out-of-Fold (OOF) Accuracy: ~98.47%

These results demonstrate strong predictive capability and stable model performance.

---

## 📐 Model Architecture

```
                    ┌──────────────────────┐
                    │      Raw Dataset     │
                    │  TRAIN.csv / TEST.csv│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Engineering │
                    │  • Log Transformation │
                    │  • Statistical Stats  │
                    │  • Row Aggregations   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Feature Interactions │
                    │  • Manual Interactions│
                    │  • Auto Interactions  │
                    │    (Top Feature Pairs)│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Stratified 5-Fold CV │
                    │ Balanced Data Splits │
                    └───────┬─────┬───────┘
                            │     │
            ┌───────────────┘     └───────────────┐
            ▼                                     ▼
   ┌────────────────┐                    ┌────────────────┐
   │   LightGBM     │                    │    XGBoost     │
   │ Gradient Boost │                    │ Gradient Boost │
   └────────┬───────┘                    └────────┬───────┘
            │                                     │
            ▼                                     ▼
                  ┌───────────────────────────┐
                  │        CatBoost           │
                  │ Gradient Boosting Model   │
                  └───────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Weighted Ensemble    │
                    │ 0.45 LGBM            │
                    │ 0.35 XGB             │
                    │ 0.20 CAT             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  OOF Evaluation      │
                    │  CV Accuracy         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Final Test Prediction│
                    │  FINAL.csv Output    │
                    └──────────────────────┘
```

---

## 📁 Dataset

The pipeline works on two CSV files:
- `TRAIN.csv` – training data with features and a `Class` target column
- `TEST.csv` – test data for final prediction (contains an `ID` column)

The `ID` column from `TEST.csv` is preserved for submission generation.

## 🔧 Data Loading and Preprocessing

```python
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")

X = train.drop("Class", axis=1)
y = train["Class"]
X_test = test.drop("ID", axis=1)
```

## 🧠 Feature Engineering

Feature engineering generates additional informative features from the raw dataset to help models learn statistical patterns.

### Row-wise statistics
For each sample, compute:
- mean, std, median, max, min, range, sum
- skew, kurtosis, 25th and 75th percentiles
- coefficient of variation (cv)

### Log transformation
A log transform is applied to feature `F38`:
```python
log_F38 = np.log1p(np.abs(F38))
```
Helps reduce skewness and stabilize variance.

### Feature Interactions
Interactions capture relationships between pairs of variables.

#### Manual interactions
Example pairs:
- `F01 * F02`
- `F10 * F11`
- `F20 * F21`

#### Automatic interactions
1. Train a temporary LightGBM model on the data.
2. Extract feature importances and select the top 10 features.
3. Create all pairwise products (45 combinations) among them.

Only top features are used to avoid feature explosion.

## 🔄 Cross Validation Strategy

Stratified 5-fold cross-validation (`StratifiedKFold(n_splits=5)`) ensures class distribution is maintained in each fold.

Workflow:
1. Split data into 5 stratified folds.
2. For each iteration, train on 4 folds and validate on the remaining fold.
3. Collect OOF predictions for each sample.

## 🎯 Machine Learning Models

Three gradient boosting models are trained independently on each fold:

### 1. LightGBM
- `n_estimators=1500`
- `learning_rate=0.02`
- `num_leaves=64`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`

LightGBM is chosen for its speed, efficiency, and strong performance on tabular data.

### 2. XGBoost
Chosen for strong regularization and ability to capture complex feature interactions.

### 3. CatBoost
Used to increase model diversity and improve generalization.

## ⚖️ Ensemble Learning

Rather than relying on a single model, predictions from all three are combined via weighted averaging:

```
Final prediction = 0.45 * LGBM + 0.35 * XGB + 0.20 * CAT
```

This ensemble reduces variance, improves stability, and raises accuracy.

## 📊 Out-of-Fold Evaluation

During CV, each fold's validation predictions form the OOF set. These are used to calculate:
- Fold accuracy
- Mean CV accuracy
- OOF accuracy (realistic estimate of unseen performance)

## 🚀 Final Prediction Generation

After CV training:
1. Each model predicts on `X_test` to produce `pred_lgb`, `pred_xgb`, `pred_cat`.
2. Combine using ensemble weights.
3. Threshold at 0.5 to assign class labels.

## 📝 Submission

Generate `FINAL.csv` with the following structure:

```
ID,Class
<sample_id>,<0 or 1>
```

This file can be submitted for evaluation or competitions.

## 📈 Model Performance

- Mean Cross-Validation Accuracy ≈ **98.47%**
- OOF Accuracy ≈ **98.47%**

Consistency across folds indicates strong generalization and reliability.

## 🛠️ Tools & Libraries

- Python (pandas, numpy)
- LightGBM
- XGBoost
- CatBoost
- scikit-learn (StratifiedKFold)

## ⚙️ Setup & Dependencies

The notebook (`1.ipynb`) starts by installing the required libraries using pip:

```python
!pip install catboost
!pip install lightgbm
!pip install xgboost
```

Ensure these packages are available in the selected Python environment before running the workflow. You can also install them manually in a terminal:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

### Kernel and Paths

- Use a Python kernel with GPU support if you wish to train XGBoost on CUDA; the notebook configures `device="cuda"` and `tree_method="hist"` for XGBoost.
- File paths in the code are absolute (e.g., `"C:\Users\shivam tyagi\OneDrive\Desktop\model\TRAIN.csv"`). Adjust them or convert to relative paths depending on where the dataset resides.
- The notebook originally read `TRAIN.xlsx` by mistake; make sure your training data is in CSV format or update the path accordingly.

---

---

*This README outlines the structure and methodology of the binary classification pipeline. Adjust hyperparameters and feature engineering steps as needed for your specific dataset.*
