"""Comprobación independiente de predicciones y métricas guardadas."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .entrenar import BASE, cargar_datos
from .preprocesamiento import preparar_documentos


def verificar() -> None:
    resumen = json.loads((BASE / "outputs" / "resumen_ejecucion.json").read_text(encoding="utf-8"))
    artifact = joblib.load(BASE / "models" / "clasificador_tfidf.joblib")
    _, test, _, _ = cargar_datos()
    entrada = preparar_documentos(test.text, lematizar=artifact["lemmatize"])
    X = artifact["vectorizer"].transform(entrada)
    pred = artifact["classifier"].predict(X)
    almacenadas = pd.read_csv(BASE / "outputs" / "predicciones_test.csv")
    if len(almacenadas) != len(test) or not np.array_equal(pred, almacenadas.prediccion.to_numpy()):
        raise AssertionError("Las predicciones no coinciden con las guardadas")
    accuracy = accuracy_score(test.label, pred)
    f1 = f1_score(test.label, pred, average="macro")
    if abs(accuracy - resumen["accuracy_test"]) > 1e-10 or abs(f1 - resumen["f1_macro_test"]) > 1e-10:
        raise AssertionError("Métricas diferentes de las registradas")
    print(f"VALIDADO: {len(test)} noticias | Accuracy={accuracy:.4f} | F1-macro={f1:.4f}")


if __name__ == "__main__":
    verificar()
