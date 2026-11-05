"""Comparación supervisada TF-IDF y evaluación final sobre AG News."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .preprocesamiento import preparar_documentos, stopwords_ingles

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
ETIQUETAS = ("Business", "Sci_Tech", "Sports", "World")
SEMILLA = 42


def encontrar_dato(nombre: str) -> Path:
    for carpeta in (ROOT / "data", ROOT / "data" / "ag_news"):
        ruta = carpeta / nombre
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(f"No se encontró {nombre} dentro de data/ o data/ag_news/")


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    archivo_train, archivo_test = (encontrar_dato(f"ag_news_{tipo}.csv") for tipo in ("train", "test"))
    train, test = pd.read_csv(archivo_train), pd.read_csv(archivo_test)
    for nombre, tabla in (("train", train), ("test", test)):
        if list(tabla.columns) != ["text", "label"]:
            raise ValueError(f"{nombre}: se esperan las columnas text,label")
        if tabla[["text", "label"]].isna().any().any():
            raise ValueError(f"{nombre}: se detectaron valores nulos")
        if not set(tabla["label"]).issubset(ETIQUETAS):
            raise ValueError(f"{nombre}: aparecen etiquetas inesperadas")
    if set(train.text) & set(test.text):
        raise ValueError("Se detectaron textos idénticos entre entrenamiento y prueba")
    return train, test, archivo_train, archivo_test


def configuraciones() -> list[dict]:
    return [
        dict(nombre="Naive Bayes | unigramas", algoritmo="NB", max_features=10000, ngram_range=(1, 1)),
        dict(nombre="Regresión logística | unigramas", algoritmo="LR", max_features=15000, ngram_range=(1, 1)),
        dict(nombre="SVM lineal | unigramas", algoritmo="SVM", max_features=15000, ngram_range=(1, 1)),
        dict(nombre="Naive Bayes | unigramas y bigramas", algoritmo="NB", max_features=15000, ngram_range=(1, 2)),
        dict(nombre="Regresión logística | unigramas y bigramas", algoritmo="LR", max_features=15000, ngram_range=(1, 2)),
        dict(nombre="SVM lineal | unigramas y bigramas", algoritmo="SVM", max_features=15000, ngram_range=(1, 2)),
    ]


def crear_vectorizador(config: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=int(config["max_features"]),
        ngram_range=tuple(config["ngram_range"]),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=stopwords_ingles(),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        dtype=np.float32,
    )


def crear_modelo(nombre: str):
    if nombre == "NB":
        return MultinomialNB(alpha=0.5)
    if nombre == "LR":
        return LogisticRegression(C=2.0, max_iter=1500, solver="lbfgs", random_state=SEMILLA)
    if nombre == "SVM":
        return LinearSVC(C=1.0, dual="auto", random_state=SEMILLA)
    raise ValueError(f"Modelo desconocido: {nombre}")


def evitar_duplicados_en_validacion(indices_train, indices_val, documentos, etiquetas):
    """Agrupa textos normalizados idénticos sin alterar tamaño ni balance del split."""
    train_ids, val_ids = set(map(int, indices_train)), set(map(int, indices_val))
    etiqueta = list(etiquetas)
    while True:
        textos_train = {documentos[i] for i in train_ids}
        repetido = next((i for i in sorted(val_ids) if documentos[i] in textos_train), None)
        if repetido is None:
            break
        textos_val = {documentos[i] for i in val_ids}
        conteos = {}
        for doc in documentos:
            conteos[doc] = conteos.get(doc, 0) + 1
        reemplazo = next((i for i in sorted(train_ids)
                          if etiqueta[i] == etiqueta[repetido]
                          and documentos[i] not in textos_val
                          and conteos[documentos[i]] == 1), None)
        if reemplazo is None:
            raise RuntimeError("No se encontró un reemplazo para mantener duplicados fuera de validación")
        val_ids.remove(repetido)
        train_ids.add(repetido)
        train_ids.remove(reemplazo)
        val_ids.add(reemplazo)
    return np.array(sorted(train_ids)), np.array(sorted(val_ids))


def dibujar_comparacion(tabla: pd.DataFrame, destino: Path) -> None:
    datos = tabla.sort_values("f1_macro_val")
    fig, ax = plt.subplots(figsize=(11, 5))
    barras = ax.barh(datos.nombre, datos.f1_macro_val, alpha=0.9)
    for b, resultado in zip(barras, datos.f1_macro_val):
        ax.text(min(0.99, b.get_width() + 0.005), b.get_y() + b.get_height()/2,
                f"{resultado:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("F1-macro en validación")
    ax.set_title("AG News | Comparación de modelos y representaciones")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def dibujar_confusion(matriz: np.ndarray, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(ETIQUETAS)), ETIQUETAS, rotation=30, ha="right")
    ax.set_yticks(range(len(ETIQUETAS)), ETIQUETAS)
    ax.set_xlabel("Categoría predicha")
    ax.set_ylabel("Categoría real")
    ax.set_title("AG News | Matriz de confusión (test)")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > matriz.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destino, dpi=170)
    plt.close(fig)


def huella_archivo(archivo: Path) -> str:
    return hashlib.sha256(archivo.read_bytes()).hexdigest()


def ejecutar(*, lematizar: bool = False) -> dict:
    BASE.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    BASE.joinpath("models").mkdir(parents=True, exist_ok=True)
    train, test, train_csv, test_csv = cargar_datos()
    print(f"AG News: train={len(train)}, test reservado={len(test)}", flush=True)
    print("Preprocesamiento: normalización + SpaCy " + ("con lematización" if lematizar else "word-level"), flush=True)

    # La tokenización es determinista y no aprende vocabulario de ninguna partición.
    textos_train = preparar_documentos(train.text, lematizar=lematizar)
    indices_train, indices_val = train_test_split(
        np.arange(len(train)), test_size=0.20,
        stratify=train.label, random_state=SEMILLA,
    )
    indices_train, indices_val = evitar_duplicados_en_validacion(
        indices_train, indices_val, textos_train, train.label
    )
    docs_train = [textos_train[i] for i in indices_train]
    docs_val = [textos_train[i] for i in indices_val]
    assert not set(docs_train).intersection(docs_val), "Duplicados normalizados entre train y val"
    y_train = train.label.iloc[indices_train].tolist()
    y_val = train.label.iloc[indices_val].tolist()
    resultados = []

    # Ningún vectorizador se ajusta utilizando validación ni prueba.
    for cfg in configuraciones():
        vec = crear_vectorizador(cfg)
        X_tr = vec.fit_transform(docs_train)
        X_val = vec.transform(docs_val)
        modelo = crear_modelo(cfg["algoritmo"])
        modelo.fit(X_tr, y_train)
        preds = modelo.predict(X_val)
        fila = {
            **cfg,
            "ngram_range": str(cfg["ngram_range"]),
            "vocabulario_real": int(len(vec.vocabulary_)),
            "accuracy_val": float(accuracy_score(y_val, preds)),
            "f1_macro_val": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        }
        resultados.append(fila)
        print(f"{cfg['nombre']}: val F1={fila['f1_macro_val']:.4f} | Accuracy={fila['accuracy_val']:.4f}", flush=True)

    comparacion = pd.DataFrame(resultados).sort_values(
        ["f1_macro_val", "accuracy_val", "nombre"], ascending=[False, False, True],
    ).reset_index(drop=True)
    comparacion.to_csv(BASE / "outputs" / "comparacion_validacion.csv", index=False, encoding="utf-8-sig")
    dibujar_comparacion(comparacion, BASE / "outputs" / "comparacion_modelos.png")

    ganador_nombre = comparacion.iloc[0]["nombre"]
    elegido = next(cfg for cfg in configuraciones() if cfg["nombre"] == ganador_nombre)
    print(f"Seleccionado por F1-macro de validación: {ganador_nombre}", flush=True)

    # Se reentrena con todo el train de la cátedra después de seleccionar los hiperparámetros.
    vectorizador_final = crear_vectorizador(elegido)
    X_train_completo = vectorizador_final.fit_transform(textos_train)
    modelo_final = crear_modelo(elegido["algoritmo"])
    modelo_final.fit(X_train_completo, train.label.tolist())

    # El test se transforma y evalúa una única vez; no participa en la selección.
    textos_test = preparar_documentos(test.text, lematizar=lematizar)
    X_test = vectorizador_final.transform(textos_test)
    y_test = test.label.tolist()
    predicciones = modelo_final.predict(X_test).tolist()
    reporte = classification_report(y_test, predicciones, labels=ETIQUETAS,
                                    output_dict=True, zero_division=0)
    reporte_txt = classification_report(y_test, predicciones, labels=ETIQUETAS,
                                        zero_division=0, digits=4)
    matriz = confusion_matrix(y_test, predicciones, labels=ETIQUETAS)
    dibujar_confusion(matriz, BASE / "outputs" / "matriz_confusion.png")
    pd.DataFrame(reporte).T.to_csv(BASE / "outputs" / "reporte_clasificacion.csv", encoding="utf-8-sig")
    pd.DataFrame(matriz, columns=ETIQUETAS, index=ETIQUETAS).to_csv(
        BASE / "outputs" / "matriz_confusion.csv", encoding="utf-8-sig"
    )
    (BASE / "outputs" / "classification_report.txt").write_text(reporte_txt, encoding="utf-8")
    pd.DataFrame({
        "fila_test": np.arange(len(test)),
        "real": y_test,
        "prediccion": predicciones,
        "correcto": [r == p for r, p in zip(y_test, predicciones)],
    }).to_csv(BASE / "outputs" / "predicciones_test.csv", index=False, encoding="utf-8-sig")

    confusiones = []
    for i, real in enumerate(ETIQUETAS):
        for j, pred in enumerate(ETIQUETAS):
            if i != j and matriz[i, j] > 0:
                confusiones.append({"real": real, "prediccion": pred, "cantidad": int(matriz[i, j])})
    confusiones.sort(key=lambda t: t["cantidad"], reverse=True)
    pd.DataFrame(confusiones).to_csv(BASE / "outputs" / "confusiones.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "vectorizer": vectorizador_final,
        "classifier": modelo_final,
        "classes": ETIQUETAS,
        "lemmatize": lematizar,
    }
    joblib.dump(artifact, BASE / "models" / "clasificador_tfidf.joblib", compress=3)
    resumen = {
        "dataset": "AG News (muestra oficial de la cátedra)",
        "train_catedra": int(len(train)), "test_catedra": int(len(test)),
        "train_desarrollo": int(len(indices_train)),
        "validacion": int(len(indices_val)),
        "semilla": SEMILLA,
        "lematizacion_spacy": bool(lematizar),
        "spacy_modelo": "en_core_web_sm" if lematizar else "spacy.blank('en')",
        "modelo_seleccionado": elegido,
        "f1_macro_validacion": float(comparacion.iloc[0]["f1_macro_val"]),
        "vocabulario_final": int(len(vectorizador_final.vocabulary_)),
        "accuracy_test": float(accuracy_score(y_test, predicciones)),
        "f1_macro_test": float(f1_score(y_test, predicciones, average="macro", zero_division=0)),
        "metricas_por_clase": {label: reporte[label] for label in ETIQUETAS},
        "principales_confusiones": confusiones[:6],
        "sha256_train": huella_archivo(train_csv),
        "sha256_test": huella_archivo(test_csv),
        "versiones": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "spacy": spacy.__version__,
        },
    }
    (BASE / "outputs" / "resumen_ejecucion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCLASIFICACIÓN FINAL\n" + reporte_txt, flush=True)
    print(f"Accuracy test={resumen['accuracy_test']:.4f} | F1-macro test={resumen['f1_macro_test']:.4f}", flush=True)
    print("Resultados guardados en checkpoint_3_tfidf/outputs/", flush=True)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar benchmark TF-IDF AG News")
    parser.add_argument("--lematizar", action="store_true", help="Usar en_core_web_sm (requiere descarga previa)")
    args = parser.parse_args()
    ejecutar(lematizar=args.lematizar)


if __name__ == "__main__":
    main()
