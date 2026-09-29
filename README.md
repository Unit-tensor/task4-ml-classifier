# Task 4 - First Machine Learning Classifier

A supervised classification pipeline built with **Scikit-Learn** that predicts Titanic passenger survival using a Decision Tree, compared against a Logistic Regression baseline.

## Objectives
- Train/test split
- Model training with Scikit-Learn
- Evaluation metrics: Accuracy, Confusion Matrix (plus precision/recall/F1)

## Project structure
```
task4-ml-classifier/
├── train.py
├── requirements.txt
├── model.joblib              # saved pipeline (created by train.py)
└── outputs/
    ├── metrics.txt
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── decision_tree.png
```

## How to run
```bash
pip install -r requirements.txt
python train.py
```

## Pipeline
1. **Features**: `pclass, age, sibsp, parch, fare, sex, embarked` -> target `survived`.
2. **Split**: 80/20 stratified train/test split (`random_state=42`).
3. **Preprocessing** (`ColumnTransformer`): median imputation for numbers, most-frequent imputation and one-hot encoding for categories. It is fitted on the training set only, so there is no data leakage.
4. **Model**: `DecisionTreeClassifier` tuned with 5-fold `GridSearchCV` (`max_depth`, `min_samples_leaf`) to limit overfitting.
5. **Baseline**: Logistic Regression for comparison.
6. **Evaluation**: accuracy, confusion matrix, classification report, feature importance.

## Results
Exact numbers are written to `outputs/metrics.txt` on each run. Expect roughly 80% accuracy, with `sex`, `fare`, `pclass` and `age` as the most important features.

## Skills demonstrated
Scikit-Learn pipelines, hyperparameter tuning, leakage-safe preprocessing, model evaluation and persistence with joblib.
