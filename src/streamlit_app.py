"""
streamlit_app.py

Aplicación Streamlit para el Proyecto Integrador M5.

Incluye:
- Predicción individual de pago a tiempo.
- Predicción batch desde archivo CSV.
- Visualización de métricas de data drift.
- Comparación de distribuciones históricas vs recientes.
- Recomendaciones automáticas según alertas de drift.

Ejecución desde la raíz del proyecto:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from ft_engineering import TARGET, prepare_dataset  # noqa: E402
from model_monitoring import load_and_split_data  # noqa: E402

DATA_PATH = PROJECT_ROOT / "Base_de_datos.csv"
MODEL_PATH = PROJECT_ROOT / "modelo_pago_atiempo.pkl"
DRIFT_CSV_PATH = PROJECT_ROOT / "drift_report.csv"
DRIFT_JSON_PATH = PROJECT_ROOT / "drift_report.json"
DRIFT_HISTORY_PATH = PROJECT_ROOT / "drift_history.csv"

REQUIRED_COLUMNS = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
]


st.set_page_config(
    page_title="PIM5 - Scoring crediticio",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Carga la base original del proyecto."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Carga el pipeline entrenado."""
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def get_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Obtiene probabilidad de clase positiva Pago_atiempo = 1."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    prediction = model.predict(X)
    return prediction.astype(float)


def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la misma ingeniería de características usada en entrenamiento."""
    df = df.copy()
    df_prepared = prepare_dataset(df)

    if TARGET in df_prepared.columns:
        df_prepared = df_prepared.drop(columns=[TARGET])

    return df_prepared


def validate_columns(df: pd.DataFrame) -> List[str]:
    """Devuelve columnas obligatorias faltantes."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def format_money(value: float) -> str:
    """Da formato simple a valores monetarios."""
    return f"${value:,.0f}"


def build_input_row(reference: pd.DataFrame) -> Dict[str, object]:
    """Construye formulario para predicción individual."""
    st.subheader("Predicción individual")

    c1, c2, c3 = st.columns(3)

    tipo_credito_values = sorted(reference["tipo_credito"].dropna().unique().tolist())
    tipo_laboral_values = sorted(reference["tipo_laboral"].dropna().astype(str).unique().tolist())
    tendencia_values = sorted(reference["tendencia_ingresos"].dropna().astype(str).unique().tolist())

    with c1:
        tipo_credito = st.selectbox("Tipo de crédito", tipo_credito_values)
        fecha_prestamo = st.date_input("Fecha del préstamo")
        capital_prestado = st.number_input("Capital prestado", min_value=0.0, value=float(reference["capital_prestado"].median()), step=100000.0)
        plazo_meses = st.number_input("Plazo en meses", min_value=1, value=int(reference["plazo_meses"].median()), step=1)
        edad_cliente = st.number_input("Edad del cliente", min_value=18, max_value=100, value=int(reference["edad_cliente"].median()), step=1)
        tipo_laboral = st.selectbox("Tipo laboral", tipo_laboral_values)
        salario_cliente = st.number_input("Salario del cliente", min_value=0, value=int(reference["salario_cliente"].median()), step=100000)

    with c2:
        total_otros_prestamos = st.number_input("Total otros préstamos", min_value=0, value=int(reference["total_otros_prestamos"].median()), step=100000)
        cuota_pactada = st.number_input("Cuota pactada", min_value=0, value=int(reference["cuota_pactada"].median()), step=10000)
        puntaje = st.number_input("Puntaje interno", min_value=0.0, max_value=100.0, value=float(reference["puntaje"].median()), step=1.0)
        puntaje_datacredito = st.number_input("Puntaje DataCrédito", min_value=0.0, value=float(reference["puntaje_datacredito"].median()), step=1.0)
        cant_creditosvigentes = st.number_input("Créditos vigentes", min_value=0, value=int(reference["cant_creditosvigentes"].median()), step=1)
        huella_consulta = st.number_input("Huellas de consulta", min_value=0, value=int(reference["huella_consulta"].median()), step=1)
        tendencia_ingresos = st.selectbox("Tendencia de ingresos", tendencia_values)

    with c3:
        saldo_mora = st.number_input("Saldo en mora", min_value=0.0, value=float(reference["saldo_mora"].median()), step=10000.0)
        saldo_total = st.number_input("Saldo total", min_value=0.0, value=float(reference["saldo_total"].median()), step=10000.0)
        saldo_principal = st.number_input("Saldo principal", min_value=0.0, value=float(reference["saldo_principal"].median()), step=10000.0)
        saldo_mora_codeudor = st.number_input("Saldo mora codeudor", min_value=0.0, value=float(reference["saldo_mora_codeudor"].median()), step=10000.0)
        creditos_sector_financiero = st.number_input("Créditos sector financiero", min_value=0, value=int(reference["creditos_sectorFinanciero"].median()), step=1)
        creditos_sector_cooperativo = st.number_input("Créditos sector cooperativo", min_value=0, value=int(reference["creditos_sectorCooperativo"].median()), step=1)
        creditos_sector_real = st.number_input("Créditos sector real", min_value=0, value=int(reference["creditos_sectorReal"].median()), step=1)
        promedio_ingresos = st.number_input("Promedio ingresos DataCrédito", min_value=0.0, value=float(reference["promedio_ingresos_datacredito"].median()), step=10000.0)

    return {
        "tipo_credito": tipo_credito,
        "fecha_prestamo": str(fecha_prestamo),
        "capital_prestado": capital_prestado,
        "plazo_meses": plazo_meses,
        "edad_cliente": edad_cliente,
        "tipo_laboral": tipo_laboral,
        "salario_cliente": salario_cliente,
        "total_otros_prestamos": total_otros_prestamos,
        "cuota_pactada": cuota_pactada,
        "puntaje": puntaje,
        "puntaje_datacredito": puntaje_datacredito,
        "cant_creditosvigentes": cant_creditosvigentes,
        "huella_consulta": huella_consulta,
        "saldo_mora": saldo_mora,
        "saldo_total": saldo_total,
        "saldo_principal": saldo_principal,
        "saldo_mora_codeudor": saldo_mora_codeudor,
        "creditos_sectorFinanciero": creditos_sector_financiero,
        "creditos_sectorCooperativo": creditos_sector_cooperativo,
        "creditos_sectorReal": creditos_sector_real,
        "promedio_ingresos_datacredito": promedio_ingresos,
        "tendencia_ingresos": tendencia_ingresos,
    }


def render_prediction_tab(reference: pd.DataFrame, model) -> None:
    """Renderiza predicción individual y batch."""
    threshold = st.slider("Umbral para clasificar Pago a tiempo", 0.0, 1.0, 0.50, 0.01)

    input_row = build_input_row(reference)

    if st.button("Predecir caso individual"):
        raw_input = pd.DataFrame([input_row])
        X_input = prepare_for_prediction(raw_input)
        probability = get_probability(model, X_input)[0]
        predicted_class = int(probability >= threshold)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probabilidad de pago a tiempo", f"{probability:.2%}")
        col_b.metric("Clasificación", "Pago a tiempo" if predicted_class == 1 else "Riesgo de atraso")
        col_c.metric("Umbral usado", f"{threshold:.2f}")

        if predicted_class == 1:
            st.success("El modelo clasifica este caso como probable pago a tiempo.")
        else:
            st.warning("El modelo clasifica este caso con riesgo de no pago a tiempo. Conviene revisión manual o política de mitigación.")

    st.divider()
    st.subheader("Predicción batch")
    st.write("Carga un CSV con las columnas originales del dataset. Puede incluir o no la columna `Pago_atiempo`.")

    uploaded_file = st.file_uploader("Archivo CSV para predicción batch", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing = validate_columns(batch_df)

        if missing:
            st.error("Faltan columnas obligatorias: " + ", ".join(missing))
            return

        X_batch = prepare_for_prediction(batch_df)
        probabilities = get_probability(model, X_batch)
        predictions = (probabilities >= threshold).astype(int)

        output = batch_df.copy()
        output["probabilidad_pago_atiempo"] = probabilities
        output["prediccion_pago_atiempo"] = predictions
        output["clasificacion"] = np.where(predictions == 1, "Pago a tiempo", "Riesgo de atraso")

        st.success("Predicción batch generada correctamente.")
        st.dataframe(output.head(100), use_container_width=True)

        st.download_button(
            label="Descargar predicciones CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_pago_atiempo.csv",
            mime="text/csv",
        )


def render_monitoring_tab() -> None:
    """Renderiza monitoreo de drift."""
    st.subheader("Monitoreo de data drift")

    if not DRIFT_CSV_PATH.exists() or not DRIFT_JSON_PATH.exists():
        st.warning("No se encontraron reportes de drift. Ejecuta primero `python src/model_monitoring.py`.")
        return

    drift_df = pd.read_csv(DRIFT_CSV_PATH)

    with open(DRIFT_JSON_PATH, "r", encoding="utf-8") as file:
        drift_json = json.load(file)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas referencia", f"{drift_json['baseline_rows']:,}")
    c2.metric("Filas recientes", f"{drift_json['current_rows']:,}")
    c3.metric("Variables con drift", len(drift_json["variables_with_drift"]))
    c4.metric("Drift detectado", "Sí" if drift_json["drift_detected"] else "No")

    if drift_json["drift_detected"]:
        st.error("Se detectó drift en: " + ", ".join(drift_json["variables_with_drift"]))
    else:
        st.success("No se detectó drift relevante según los umbrales definidos.")

    top_drift = drift_df.sort_values("psi", ascending=False).head(10)
    st.write("Top variables por PSI")
    st.bar_chart(top_drift.set_index("variable")["psi"])

    st.write("Tabla completa de métricas")
    st.dataframe(drift_df, use_container_width=True)

    st.divider()
    st.subheader("Comparación histórica vs reciente")

    baseline, current = load_and_split_data(DATA_PATH)
    variables = drift_df["variable"].tolist()
    selected_variable = st.selectbox("Variable a comparar", variables)

    if selected_variable in baseline.columns:
        if pd.api.types.is_numeric_dtype(baseline[selected_variable]):
            hist_values = baseline[selected_variable].dropna()
            curr_values = current[selected_variable].dropna()
            bins = np.histogram_bin_edges(hist_values, bins=10)

            hist_counts, _ = np.histogram(hist_values, bins=bins)
            curr_counts, _ = np.histogram(curr_values, bins=bins)

            labels = [f"{bins[i]:.1f} - {bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
            comparison = pd.DataFrame({
                "rango": labels,
                "referencia": hist_counts / max(hist_counts.sum(), 1),
                "reciente": curr_counts / max(curr_counts.sum(), 1),
            })
        else:
            hist_dist = baseline[selected_variable].fillna("SIN_DATO").astype(str).value_counts(normalize=True)
            curr_dist = current[selected_variable].fillna("SIN_DATO").astype(str).value_counts(normalize=True)
            comparison = pd.concat([hist_dist, curr_dist], axis=1).fillna(0).reset_index()
            comparison.columns = ["rango", "referencia", "reciente"]

        st.bar_chart(comparison.set_index("rango")[["referencia", "reciente"]])
        st.dataframe(comparison, use_container_width=True)

    st.divider()
    st.subheader("Análisis temporal")

    if DRIFT_HISTORY_PATH.exists():
        history = pd.read_csv(DRIFT_HISTORY_PATH)
        variable_history = st.selectbox(
            "Variable para tendencia temporal",
            sorted(history["variable"].unique().tolist()),
            key="history_variable",
        )
        filtered_history = history[history["variable"] == variable_history].copy()
        filtered_history["timestamp"] = pd.to_datetime(filtered_history["timestamp"])
        st.line_chart(filtered_history.set_index("timestamp")["psi"])
        st.dataframe(filtered_history, use_container_width=True)
    else:
        st.info("Aún no existe `drift_history.csv`. Se mostrará evolución temporal cuando el monitoreo se ejecute más de una vez con historial habilitado.")
        st.write(f"Última ejecución registrada: {drift_json['timestamp']}")

    st.divider()
    st.subheader("Recomendaciones automáticas")

    if drift_json["drift_detected"]:
        st.warning(
            "Se recomienda revisar las variables con drift, validar si el cambio responde a estacionalidad "
            "o a un cambio real de población, y evaluar reentrenamiento si el patrón se mantiene."
        )
    else:
        st.success("El modelo puede continuar operando bajo los umbrales actuales de monitoreo.")


def render_results_tab() -> None:
    """Renderiza resultados del entrenamiento."""
    st.subheader("Resultados de entrenamiento")

    results_path = PROJECT_ROOT / "resultados_modelos.csv"

    if not results_path.exists():
        st.warning("No se encontró `resultados_modelos.csv`. Ejecuta `python src/model_training_evaluation.py`.")
        return

    results = pd.read_csv(results_path)
    st.dataframe(results, use_container_width=True)

    best = results.sort_values("test_roc_auc", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor modelo", best["modelo"])
    c2.metric("F1 test", f"{best['test_f1']:.4f}")
    c3.metric("ROC-AUC test", f"{best['test_roc_auc']:.4f}")

    st.info(
        "Las métricas son muy altas. Esto puede indicar buen desempeño, pero también posible data leakage. "
        "Por eso el monitoreo y la revisión de variables son parte importante del flujo MLOps."
    )


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.title("PIM5 - Modelo de pago a tiempo")
    st.write(
        "Aplicación de apoyo para operar el modelo predictivo, revisar resultados de entrenamiento "
        "y monitorear data drift."
    )

    if not DATA_PATH.exists():
        st.error("No se encontró `Base_de_datos.csv` en la raíz del proyecto.")
        return

    if not MODEL_PATH.exists():
        st.error("No se encontró `modelo_pago_atiempo.pkl`. Ejecuta primero `python src/model_training_evaluation.py`.")
        return

    reference = load_raw_data()
    model = load_model()

    tab_prediction, tab_monitoring, tab_results = st.tabs([
        "Predicción",
        "Monitoreo de drift",
        "Resultados del modelo",
    ])

    with tab_prediction:
        render_prediction_tab(reference, model)

    with tab_monitoring:
        render_monitoring_tab()

    with tab_results:
        render_results_tab()


if __name__ == "__main__":
    main()
