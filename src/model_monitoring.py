"""
model_monitoring.py

Monitoreo de data drift para el Proyecto Integrador M5.

Este script compara una muestra de referencia contra una muestra reciente
del dataset para detectar cambios en la distribución de las variables.
Genera reportes en JSON y CSV con métricas de drift.
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ft_engineering import TARGET, DATE_COL, prepare_dataset


DATA_PATH = "Base_de_datos.csv"
JSON_REPORT_PATH = "drift_report.json"
CSV_REPORT_PATH = "drift_report.csv"
PSI_THRESHOLD = 0.25
KS_THRESHOLD = 0.20


def load_and_split_data(path: str | Path = DATA_PATH, recent_size: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la base, ordena por fecha y separa referencia vs datos recientes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    raw = pd.read_csv(path)

    if DATE_COL in raw.columns:
        raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
        raw = raw.sort_values(DATE_COL)

    prepared = prepare_dataset(raw)
    split_index = int(len(prepared) * (1 - recent_size))

    baseline = prepared.iloc[:split_index].copy()
    current = prepared.iloc[split_index:].copy()

    return baseline, current


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calcula Population Stability Index para variables numéricas."""
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if expected.empty or actual.empty:
        return 0.0

    if expected.nunique() <= 1:
        return 0.0

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))

    if len(breakpoints) <= 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / max(expected_counts.sum(), 1)
    actual_percents = actual_counts / max(actual_counts.sum(), 1)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return float(psi)


def calculate_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Calcula PSI para variables categóricas usando proporciones por categoría."""
    expected = expected.fillna("SIN_DATO").astype(str)
    actual = actual.fillna("SIN_DATO").astype(str)

    categories = sorted(set(expected.unique()) | set(actual.unique()))

    expected_dist = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual_dist = actual.value_counts(normalize=True).reindex(categories, fill_value=0)

    expected_percents = np.where(expected_dist.values == 0, 0.0001, expected_dist.values)
    actual_percents = np.where(actual_dist.values == 0, 0.0001, actual_dist.values)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return float(psi)


def get_monitoring_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifica columnas numéricas y categóricas a monitorear."""
    features = df.drop(columns=[TARGET], errors="ignore")
    numeric_cols = features.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    return numeric_cols, categorical_cols


def monitor_numeric_variable(var: str, baseline: pd.DataFrame, current: pd.DataFrame) -> Dict[str, object]:
    """Calcula métricas de drift para una variable numérica."""
    expected = baseline[var].dropna()
    actual = current[var].dropna()

    psi = calculate_psi(expected.values, actual.values)

    if expected.empty or actual.empty or expected.nunique() <= 1:
        ks_statistic = 0.0
        ks_pvalue = 1.0
    else:
        ks_statistic, ks_pvalue = ks_2samp(expected, actual)

    drift_detected = bool(psi > PSI_THRESHOLD or ks_statistic > KS_THRESHOLD)

    return {
        "variable": var,
        "tipo": "numerica",
        "psi": round(float(psi), 4),
        "ks_statistic": round(float(ks_statistic), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "drift_detected": drift_detected
    }


def monitor_categorical_variable(var: str, baseline: pd.DataFrame, current: pd.DataFrame) -> Dict[str, object]:
    """Calcula drift para una variable categórica usando PSI."""
    psi = calculate_categorical_psi(baseline[var], current[var])
    drift_detected = bool(psi > PSI_THRESHOLD)

    return {
        "variable": var,
        "tipo": "categorica",
        "psi": round(float(psi), 4),
        "ks_statistic": None,
        "ks_pvalue": None,
        "drift_detected": drift_detected
    }


def monitor_drift(
    data_path: str | Path = DATA_PATH,
    json_report_path: str | Path = JSON_REPORT_PATH,
    csv_report_path: str | Path = CSV_REPORT_PATH
) -> Dict[str, object]:
    """Ejecuta el monitoreo completo y guarda los reportes."""
    baseline, current = load_and_split_data(data_path)
    numeric_cols, categorical_cols = get_monitoring_columns(baseline)

    rows = []

    for col in numeric_cols:
        rows.append(monitor_numeric_variable(col, baseline, current))

    for col in categorical_cols:
        rows.append(monitor_categorical_variable(col, baseline, current))

    report_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    variables_with_drift = report_df.loc[report_df["drift_detected"], "variable"].tolist()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline_rows": int(len(baseline)),
        "current_rows": int(len(current)),
        "psi_threshold": PSI_THRESHOLD,
        "ks_threshold": KS_THRESHOLD,
        "drift_detected": bool(len(variables_with_drift) > 0),
        "variables_with_drift": variables_with_drift,
        "metrics": report_df.to_dict(orient="records")
    }

    report_df.to_csv(csv_report_path, index=False)

    with open(json_report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    drift_report = monitor_drift()

    print("Monitoreo de data drift finalizado.")
    print(f"Filas de referencia: {drift_report['baseline_rows']}")
    print(f"Filas recientes: {drift_report['current_rows']}")
    print(f"Drift detectado: {drift_report['drift_detected']}")

    if drift_report["variables_with_drift"]:
        print("Variables con drift:")
        for variable in drift_report["variables_with_drift"]:
            print(f"- {variable}")
    else:
        print("No se detectaron variables con drift relevante.")

    print(f"Reporte JSON guardado en: {JSON_REPORT_PATH}")
    print(f"Reporte CSV guardado en: {CSV_REPORT_PATH}")