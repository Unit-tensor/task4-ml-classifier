"""Task 4 - First ML classifier: predict Titanic survival with a Decision Tree."""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

# 1. Data ------------------------------------------------------------------
df = sns.load_dataset("titanic")
NUM = ["pclass", "age", "sibsp", "parch", "fare"]
CAT = ["sex", "embarked"]
X, y = df[NUM + CAT], df["survived"]

# 2. Train/test split (stratified keeps class balance) ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 3. Preprocessing pipeline (fit on train only -> no data leakage) ----------
def preprocessor(scale=False):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT)])

# 4. Models -----------------------------------------------------------------
tree = Pipeline([("prep", preprocessor()),
                 ("clf", DecisionTreeClassifier(random_state=42))])
grid = GridSearchCV(tree, {"clf__max_depth": [2, 3, 4, 5, 6, 8],
                           "clf__min_samples_leaf": [1, 5, 10]}, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_, f"(CV acc {grid.best_score_:.3f})")

baseline = Pipeline([("prep", preprocessor(scale=True)),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)

# 5. Evaluation ---------------------------------------------------------------
results = {}
for name, model in {"Decision Tree": best, "Logistic Regression (baseline)": baseline}.items():
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print(f"\n{name}: accuracy = {results[name]:.3f}")
print("\nClassification report (Decision Tree):")
report = classification_report(y_test, best.predict(X_test), target_names=["Died", "Survived"])
print(report)

cm = confusion_matrix(y_test, best.predict(X_test))
print("Confusion matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"]).plot(cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

# Feature importance
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["clf"].feature_importances_, index=names).sort_values()
imp.plot.barh(color="#2E86AB", figsize=(7, 5)); plt.title("Feature Importance")
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

# Tree diagram
plt.figure(figsize=(18, 8))
plot_tree(best.named_steps["clf"], feature_names=names, class_names=["Died", "Survived"],
          filled=True, rounded=True, max_depth=3, fontsize=8)
plt.savefig(OUT / "decision_tree.png", dpi=120); plt.close()

# 6. Save model + metrics ----------------------------------------------------
joblib.dump(best, ROOT / "model.joblib")
(OUT / "metrics.txt").write_text(
    f"Best params: {grid.best_params_}\n"
    + "\n".join(f"{k}: accuracy {v:.3f}" for k, v in results.items())
    + f"\n\nConfusion matrix (rows=actual, cols=predicted):\n{cm}\n\n{report}")
print("Saved model.joblib and outputs/")
