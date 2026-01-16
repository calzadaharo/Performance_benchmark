import kagglehub
import optuna
import time


from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


lf = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "neurocipher/heartdisease",
    "Heart_Disease_Prediction.csv",
)

target_column = "Heart Disease"
categorical_columns = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Number of vessels fluro",
    "Thallium",
]

numerical_columns = [
    col for col in lf.columns if col not in categorical_columns and col != target_column
]

X = lf.select(categorical_columns + numerical_columns).collect().to_numpy()
# Flatten to 1D array as required by scikit-learn
y = lf.select(target_column).collect().to_numpy().ravel()


def objective(trial, X, y):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_float("max_features", 0.1, 1.0)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    score = cross_val_score(model, X, y, cv=5).mean()

    return score


study = optuna.create_study(direction="maximize")

start_time = time.time()
study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print(study.best_trial.params)
print(study.best_trial.value)
