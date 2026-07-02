"""
model_training_evaluation.py

Entrenamiento y evaluación de modelos supervisados para el Proyecto Integrador M5.

Este script compara modelos de clasificación usando pipelines de Scikit-Learn,
selecciona el mejor modelo según ROC-AUC y guarda el pipeline final entrenado.
"""

from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ft_engineering import TARGET, load_data, prepare_dataset, split_features_target


RANDOM_STATE = 42
MODEL_PATH = "modelo_pago_atiempo.pkl"
RESULTS_PATH = "resultados_modelos.csv"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construye el preprocesador según los tipos de variables."""
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def build_models() -> List[Tuple[str, object]]:
    """Define los modelos candidatos para comparación."""
    return [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("DecisionTree", DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=80, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
    ]


def summarize_classification(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Calcula métricas principales de clasificación."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_evaluate(
    data_path: str | Path = "Base_de_datos.csv",
    model_path: str | Path = MODEL_PATH,
    results_path: str | Path = RESULTS_PATH
) -> Tuple[Pipeline, pd.DataFrame]:
    """Entrena, compara modelos y guarda el mejor pipeline."""
    df = load_data(data_path)
    df = prepare_dataset(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    results = []
    best_pipeline = None
    best_score = -1

    for name, model in build_models():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring=["accuracy", "f1", "roc_auc"],
            return_train_score=False
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = summarize_classification(y_test, y_pred, y_proba)

        result = {
            "modelo": name,
            "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
            "cv_accuracy_std": cv_scores["test_accuracy"].std(),
            "cv_f1_mean": cv_scores["test_f1"].mean(),
            "cv_f1_std": cv_scores["test_f1"].std(),
            "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv_scores["test_roc_auc"].std(),
            **{f"test_{k}": v for k, v in test_metrics.items()}
        }

        results.append(result)

        if test_metrics["roc_auc"] > best_score:
            best_score = test_metrics["roc_auc"]
            best_pipeline = pipeline

        print(f"\nModelo: {name}")
        print(f"ROC-AUC test: {test_metrics['roc_auc']:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(results_path, index=False)

    with open(model_path, "wb") as file:
        pickle.dump(best_pipeline, file)

    print("\nResumen comparativo:")
    print(results_df)
    print(f"\nMejor modelo guardado en: {model_path}")
    print(f"Resultados guardados en: {results_path}")

    return best_pipeline, results_df


if __name__ == "__main__":
    train_and_evaluate()
