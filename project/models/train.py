from pathlib import Path

import pandas as pd
import xgboost as xgb
import joblib
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hammjeff")

# Load the prepared splits from the current working folder.
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = Xtrain.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    exclude="number"
).columns.tolist()

# Handle class imbalance.
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=1,
    eval_metric="logloss"
)

# Same search grid as the successful development cell.
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__colsample_bytree": [0.8],
    "xgbclassifier__colsample_bylevel": [0.8],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [1, 5],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log each parameter combination.
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric(
                "mean_test_score", results["mean_test_score"][i]
            )
            mlflow.log_metric(
                "std_test_score", results["std_test_score"][i]
            )

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_scoring", "f1")
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.5
    mlflow.log_param(
        "classification_threshold", classification_threshold
    )

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True, zero_division=0
    )

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save the complete preprocessing and model pipeline.
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "deployment" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="model")

    print("Best parameters:", grid_search.best_params_)
    print("\nTest classification report:")
    print(classification_report(
        ytest, y_pred_test, zero_division=0
    ))
    print(f"Model saved to {model_path}")
