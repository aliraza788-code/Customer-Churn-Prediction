import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("telco.csv")

# 2. Clean data
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# 3. Split X and y
X = df[["tenure", "MonthlyCharges", "gender", "Contract"]]
y = df["Churn"]

# 4. Column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# 5. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# -----------------------------
# MODEL 1: Logistic Regression
# -----------------------------
pipe_lr = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

params_lr = {
    "model__C": [0.1, 1, 10]
}

grid_lr = GridSearchCV(pipe_lr, params_lr, cv=3)

# -----------------------------
# MODEL 2: Random Forest
# -----------------------------
pipe_rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier())
])

params_rf = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [5, 10]
}

grid_rf = GridSearchCV(pipe_rf, params_rf, cv=3)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train both models
grid_lr.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

# 8. Predictions
pred_lr = grid_lr.predict(X_test)
pred_rf = grid_rf.predict(X_test)

# 9. Accuracy
acc_lr = accuracy_score(y_test, pred_lr)
acc_rf = accuracy_score(y_test, pred_rf)

print("Logistic Regression Accuracy:", acc_lr)
print("Random Forest Accuracy:", acc_rf)

# 10. Best model choose
if acc_rf > acc_lr:
    best_model = grid_rf.best_estimator_
    print("Best Model: Random Forest")
else:
    best_model = grid_lr.best_estimator_
    print("Best Model: Logistic Regression")

# 11. Detailed report
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# 12. Save model
joblib.dump(best_model, "model.pkl")

print("\nModel saved successfully ✅")