import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
import joblib

df = pd.read_csv(r"wine_quality_classification.csv")

features = pd.drop (columns = ["quality_label"])
target = pd["quality_label"]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])

param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=1000, random_state=42)],
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
    },
    {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
    }
]

search = GridSearchCV(
    pipeline,
    param_grid,
    cv= KFold (n_splits = 5, shuffle = True, random_state = 42),
    scoring= make_scorer(accuracy_score, greater_is_better=True),
    n_jobs=-1,
    verbose=1
)

search.fit(x_train, y_train)

best_model = search.best_estimator_

y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print (accuracy)

joblib.dump(best_model, "best_wine_quality_model.joblib")