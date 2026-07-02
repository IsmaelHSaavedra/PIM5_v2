"""
ft_engineering.py

Ingeniería de características para el Proyecto Integrador M5.

Este script prepara la base histórica de créditos para el entrenamiento
del modelo de clasificación encargado de predecir la variable Pago_atiempo.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


TARGET = "Pago_atiempo"
DATE_COL = "fecha_prestamo"
CATEGORICAL_AS_TEXT = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]


def load_data(path: str | Path = "Base_de_datos.csv") -> pd.DataFrame:
    """Carga el dataset base desde un archivo CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    return pd.read_csv(path)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica representaciones comunes de valores nulos."""
    df = df.copy()
    missing_values = ["", " ", "NA", "N/A", "nan", "NaN", "None", "null", "NULL"]
    df = df.replace(missing_values, np.nan)

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas clave al tipo de dato adecuado."""
    df = df.copy()

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    for col in CATEGORICAL_AS_TEXT:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas útiles para el modelo."""
    df = df.copy()

    if DATE_COL in df.columns:
        df["anio_prestamo"] = df[DATE_COL].dt.year
        df["mes_prestamo"] = df[DATE_COL].dt.month
        df["dia_semana_prestamo"] = df[DATE_COL].dt.dayofweek
        df["trimestre_prestamo"] = df[DATE_COL].dt.quarter
        df = df.drop(columns=[DATE_COL])

    df["cuota_sobre_salario"] = df["cuota_pactada"] / df["salario_cliente"].replace(0, np.nan)
    df["capital_sobre_salario"] = df["capital_prestado"] / df["salario_cliente"].replace(0, np.nan)
    df["saldo_sobre_capital"] = df["saldo_total"] / df["capital_prestado"].replace(0, np.nan)
    df["mora_sobre_saldo"] = df["saldo_mora"] / df["saldo_total"].replace(0, np.nan)

    df["total_creditos_reportados"] = (
        df["creditos_sectorFinanciero"]
        + df["creditos_sectorCooperativo"]
        + df["creditos_sectorReal"]
    )

    df["tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)
    df["tiene_codeudor_en_mora"] = (df["saldo_mora_codeudor"].fillna(0) > 0).astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el flujo completo de preparación de características."""
    df = normalize_missing_values(df)
    df = convert_data_types(df)
    df = create_features(df)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y variable objetivo."""
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def save_prepared_dataset(
    input_path: str | Path = "Base_de_datos.csv",
    output_path: str | Path = "Base_de_datos_preparada.csv"
) -> pd.DataFrame:
    """Genera y guarda una versión preparada del dataset."""
    df = load_data(input_path)
    df_prepared = prepare_dataset(df)
    df_prepared.to_csv(output_path, index=False)

    return df_prepared

if __name__ == "__main__":
    prepared = save_prepared_dataset()
    print("Dataset preparado correctamente.")
    print(f"Filas: {prepared.shape[0]}")
    print(f"Columnas: {prepared.shape[1]}")