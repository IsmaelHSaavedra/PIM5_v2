import pickle
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from ft_engineering import TARGET, prepare_dataset  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "modelo_pago_atiempo.pkl"

app = FastAPI(
    title="PIM5 - API de predicción de pago a tiempo",
    description="API para disponibilizar el modelo predictivo de pago a tiempo.",
    version="1.0.0"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()


class CreditInput(BaseModel):
    tipo_credito: int = Field(..., ge=0)
    fecha_prestamo: str
    capital_prestado: float = Field(..., ge=0)
    plazo_meses: int = Field(..., ge=1)
    edad_cliente: int = Field(..., ge=18)
    tipo_laboral: str
    salario_cliente: float = Field(..., ge=0)
    total_otros_prestamos: float = Field(..., ge=0)
    cuota_pactada: float = Field(..., ge=0)
    puntaje: float = Field(..., ge=0)
    puntaje_datacredito: float = Field(..., ge=0)
    cant_creditosvigentes: int = Field(..., ge=0)
    huella_consulta: int = Field(..., ge=0)
    saldo_mora: float = Field(..., ge=0)
    saldo_total: float = Field(..., ge=0)
    saldo_principal: float = Field(..., ge=0)
    saldo_mora_codeudor: float = Field(..., ge=0)
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    promedio_ingresos_datacredito: float = Field(..., ge=0)
    tendencia_ingresos: str


@app.get("/")
def root():
    return {
        "mensaje": "API activa",
        "modelo": "Pago a tiempo",
        "documentacion": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: CreditInput):
    try:
        raw_input = pd.DataFrame([data.model_dump()])
        prepared_input = prepare_dataset(raw_input)

        if TARGET in prepared_input.columns:
            prepared_input = prepared_input.drop(columns=[TARGET])

        prediction = model.predict(prepared_input)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(prepared_input)[0][1]
        else:
            probability = None

        return {
            "prediction": int(prediction),
            "classification": "Pago a tiempo" if int(prediction) == 1 else "No pago a tiempo",
            "probability_pago_atiempo": None if probability is None else round(float(probability), 6)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )