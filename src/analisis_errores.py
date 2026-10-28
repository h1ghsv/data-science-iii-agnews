"""Diagnóstico descriptivo del rendimiento final sobre AG News."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train import ROOT, LABELS, preprocess_text, locate_csv


def crear_analisis():
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_test.csv")
    test = pd.read_csv(locate_csv("data/ag_news/ag_news_test.csv"))
    esperado = {"fila_test", "label_real", "label_predicha", "confianza_modelo"}
    if not esperado.issubset(pred.columns):
        raise ValueError("Faltan columnas de predicciones_test.csv")
    if len(pred) != len(test) or pred.fila_test.duplicated().any():
        raise ValueError("Las predicciones no corresponden al test disponible")
    if pred.fila_test.min() != 0 or pred.fila_test.max() != len(test)-1:
        raise ValueError("Índices de predicción incompletos")
    pred = pred.sort_values("fila_test").reset_index(drop=True)
    if not (pred.label_real.to_numpy() == test.label.to_numpy()).all():
        raise ValueError("El orden o las etiquetas del test no coinciden")
    if not pred.confianza_modelo.between(0,1).all():
        raise ValueError("Confianzas fuera de rango")

    pred["tokens"] = test.text.map(lambda x: len(preprocess_text(x).split()))
    pred["correcta"] = pred.label_real.eq(pred.label_predicha)
    pred["grupo_longitud"] = pd.qcut(
        pred["tokens"], q=4, labels=False, duplicates="drop"
    )
    resumen_longitud = (pred.groupby("grupo_longitud", observed=True)
                        .agg(n=("correcta", "size"), aciertos=("correcta", "sum"),
                             min_tokens=("tokens", "min"), max_tokens=("tokens", "max"))
                        .reset_index())
    resumen_longitud["accuracy"] = resumen_longitud.aciertos / resumen_longitud.n
    resumen_longitud["rango"] = resumen_longitud.apply(
        lambda r: f"{int(r.min_tokens)}–{int(r.max_tokens)}", axis=1)
    resumen_clases = (pred.groupby("label_real", observed=True)
                      .agg(n=("correcta", "size"), aciertos=("correcta", "sum"))
                      .reindex(LABELS).reset_index())
    resumen_clases["accuracy"] = resumen_clases.aciertos / resumen_clases.n
    confusiones = (pred[~pred.correcta].groupby(["label_real", "label_predicha"])
                   .size().sort_values(ascending=False).rename("casos")
                   .reset_index())
    errores = pred[~pred.correcta].sort_values("confianza_modelo", ascending=False)

    destino = ROOT / "outputs"
    destino.mkdir(parents=True, exist_ok=True)
    resumen_longitud.to_csv(destino / "analisis_longitud.csv", index=False)
    resumen_clases.to_csv(destino / "analisis_categorias.csv", index=False)
    confusiones.to_csv(destino / "confusiones_frecuentes.csv", index=False)
    errores[["fila_test", "label_real", "label_predicha", "confianza_modelo", "tokens"]].head(20).to_csv(
        destino / "errores_alta_confianza.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    bars = ax.bar(resumen_clases.label_real, resumen_clases.accuracy * 100)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Aciertos por categoría")
    for bar, val in zip(bars, resumen_clases.accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val*100:.1f}%", ha="center", fontsize=9)
    ax = axs[1]
    bars = ax.bar(resumen_longitud.rango, resumen_longitud.accuracy*100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Longitud del documento (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Aciertos según longitud")
    for bar, val, n in zip(bars, resumen_longitud.accuracy, resumen_longitud.n):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val*100:.1f}% (n={n})", ha="center", fontsize=8)
    fig.suptitle("AG News · diagnóstico del modelo sobre test reservado", fontsize=13)
    fig.tight_layout()
    fig.savefig(destino / "analisis_errores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    n_errors = int((~pred.correcta).sum())
    fila_1 = confusiones.iloc[0]
    fila_2 = confusiones.iloc[1]
    bajos = resumen_clases.sort_values("accuracy").iloc[0]
    altos = resumen_clases.sort_values("accuracy").iloc[-1]
    d = {
        "documentos": int(len(pred)), "errores": n_errors,
        "accuracy": float(pred.correcta.mean()),
        "confusion_principal": {"real": fila_1.label_real, "predicha": fila_1.label_predicha, "casos": int(fila_1.casos)},
        "categoria_mas_dificil": {"categoria": bajos.label_real, "accuracy": float(bajos.accuracy)},
        "categoria_mas_facil": {"categoria": altos.label_real, "accuracy": float(altos.accuracy)}
    }
    (destino / "analisis_errores_resumen.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# Análisis de errores del clasificador AG News",
        "",
        "## Resultados del test final",
        "",
        f"Sobre {len(pred):,} noticias del test reservado, se clasificaron correctamente {int(pred.correcta.sum()):,} "
        f"y se registraron {n_errors} errores (Accuracy: {pred.correcta.mean()*100:.2f} %).",
        "",
        "### Rendimiento por categoría",
        "",
        "| Categoría | Noticias | Aciertos | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for _, r in resumen_clases.iterrows():
        lineas.append(f"| {r.label_real} | {int(r.n)} | {int(r.aciertos)} | {r.accuracy*100:.2f} % |")
    lineas += ["", "### Confusiones principales", "", "| Real | Predicha | Casos |", "|---|---|---:|"]
    for _, r in confusiones.head(8).iterrows():
        lineas.append(f"| {r.label_real} | {r.label_predicha} | {int(r.casos)} |")
    lineas += ["", "### Influencia de la longitud", "", "| Tokens | Documentos | Accuracy |", "|---|---:|---:|"]
    for _, r in resumen_longitud.iterrows():
        lineas.append(f"| {r.rango} | {int(r.n)} | {r.accuracy*100:.2f} % |")
    lineas += [
        "", "![Aciertos por categoría y longitud](../outputs/analisis_errores.png)",
        "", "### Interpretación", "",
        f"La categoría con mayor proporción de aciertos entre sus noticias fue **{altos.label_real}** "
        f"({altos.accuracy*100:.1f} % de sus ejemplos correctos). "
        f"La más difícil fue **{bajos.label_real}** ({bajos.accuracy*100:.1f} %). "
        f"La confusión más frecuente fue **{fila_1.label_real} → {fila_1.label_predicha}** "
        f"({int(fila_1.casos)} noticias). También se observaron "
        f"**{fila_2.label_real} → {fila_2.label_predicha}** ({int(fila_2.casos)} casos).",
        "",
        "La variación de aciertos por longitud es descriptiva y no demuestra causalidad. "
        "`errores_alta_confianza.csv` identifica las predicciones incorrectas más seguras "
        "para una revisión cualitativa posterior, sin repetir el texto original de las noticias.",
        "",
        "**Nota metodológica:** estos resultados provienen del test utilizado una sola vez tras seleccionar "
        "la mejor época con validación. El análisis de errores es posterior y no se utiliza "
        "para cambiar hiperparámetros ni volver a entrenar con información de test. "
        "Las divisiones por longitud se calcularon únicamente para describir los resultados.",
    ]
    (ROOT / "docs" / "analisis_errores.md").write_text("\n".join(lineas)+"\n", encoding="utf-8")
    print(f"Diagnóstico finalizado: {n_errors} errores de {len(pred)} casos; gráfico y reportes generados.")
    return d


if __name__ == "__main__":
    crear_analisis()
