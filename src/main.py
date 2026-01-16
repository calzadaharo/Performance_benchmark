import kagglehub
import optuna
import time


from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from typing import Literal

BenchmarkType = Literal["RandomForest", "Optuna", "HistGradientBoosting"]

benchmark: BenchmarkType = "Optuna"

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

if benchmark == "RandomForest":
    init_time = time.time()
    best_score = 0
    for i in range(100):
        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        score = cross_val_score(model, X, y, cv=5).mean()
        best_score = max(best_score, score)
        print(f"Model {i} trained with score {score}. Best score: {best_score}")
    end_time = time.time()
    print(f"Time taken training 100 models: {end_time - init_time} seconds")

if benchmark == "HistGradientBoosting":
    init_time = time.time()
    best_score = 0
    for i in range(100):
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=42,
        )
        score = cross_val_score(model, X, y, cv=5).mean()
        best_score = max(best_score, score)
        print(f"Model {i} trained with score {score}. Best score: {best_score}")
    end_time = time.time()
    print(f"Time taken training 100 models: {end_time - init_time} seconds")

if benchmark == "Optuna":

    def objective_rf(trial, X, y):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
            n_jobs=-1,
        )

        score = cross_val_score(model, X, y, cv=5).mean()

        return score

    def objective_hgb(trial, X, y):
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
        max_depth = trial.suggest_int("max_depth", 3, 10)

        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )

        score = cross_val_score(model, X, y, cv=5).mean()

        return score

    study_rf = optuna.create_study(direction="maximize")
    study_hgb = optuna.create_study(direction="maximize")

    rf_start_time = time.time()
    study_rf.optimize(lambda trial: objective_rf(trial, X, y), n_trials=100)
    rf_end_time = time.time()

    hgb_start_time = time.time()
    study_hgb.optimize(lambda trial: objective_hgb(trial, X, y), n_trials=100)
    hgb_end_time = time.time()

    print("--------------------------------")
    print("Results")
    print("--------------------------------")
    print("🌳 Random Forest")
    print("--------------------------------")
    print(f"Radom forest time taken: {rf_end_time - rf_start_time} seconds")
    print(f"Random Forest best parameters: {study_rf.best_trial.params}")
    print(f"Random Forest best score: {study_rf.best_trial.value}")
    print("--------------------------------")
    print("📊 Hist Gradient Boosting")
    print("--------------------------------")
    print(f"Hist Gradient Boosting time taken: {hgb_end_time - hgb_start_time} seconds")
    print(f"Hist Gradient Boosting best parameters: {study_hgb.best_trial.params}")
    print(f"Hist Gradient Boosting best score: {study_hgb.best_trial.value}")
    print("--------------------------------")
