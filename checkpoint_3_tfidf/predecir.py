"""Inferencia sobre texto nuevo con el baseline del checkpoint 3."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from .preprocesamiento import preparar_documentos

BASE = Path(__file__).resolve().parent


def predecir(texto: str, ruta_modelo: Path | None = None) -> str:
    ruta = ruta_modelo or BASE / "models" / "clasificador_tfidf.joblib"
    objeto = joblib.load(ruta)
    limpio = preparar_documentos([texto], lematizar=objeto["lemmatize"])
    matriz = objeto["vectorizer"].transform(limpio)
    return str(objeto["classifier"].predict(matriz)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificar una noticia AG News")
    parser.add_argument("texto", help="Noticia en inglés")
    args = parser.parse_args()
    print(predecir(args.texto))


if __name__ == "__main__":
    main()
