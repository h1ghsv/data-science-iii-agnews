"""Genera el informe técnico del experimento LoRA."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
REPORTES = BASE / "reportes"
DESTINO = REPORTES / "Arevalo_JuanPablo_Checkpoint_NLP3.pdf"


def fuente():
    # Busca una fuente instalada; evita depender de rutas fijas de Colab.
    from matplotlib import font_manager
    regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    pdfmetrics.registerFont(TTFont("FuenteEDA", regular))
    pdfmetrics.registerFont(TTFont("FuenteEDAB", bold))
    pdfmetrics.registerFontFamily("FuenteEDA", normal="FuenteEDA", bold="FuenteEDAB")
    return "FuenteEDA", "FuenteEDAB"


def validar(datos):
    if datos.get("ejecucion_real") is not True:
        raise ValueError("No se encontraron registros válidos del entrenamiento LoRA")
    for camino in ("matriz_confusion_lora.png", "curvas_loss.png", "classification_report_lora.csv"):
        if not (OUT / camino).is_file():
            raise FileNotFoundError(f"Falta resultado de ejecución real: {camino}")
    assert datos["dataset"]["test"] == 2000
    assert datos["num_labels"] == 4
    assert datos["lora"]["parametros"]["fraccion_entrenable"] < .03


def porcentaje(v):
    return f"{v * 100:.2f} %".replace(".", ",")


def numero(n):
    return f"{int(n):,}".replace(",", ".")


def tabla(filas, widths, header=True):
    t = Table([[str(c) for c in fila] for fila in filas], colWidths=widths, repeatRows=1 if header else 0,
              hAlign="LEFT", splitByRow=1)
    estilos = [
        ("FONTNAME", (0,0), (-1,-1), "FuenteEDA"),
        ("FONTSIZE", (0,0), (-1,-1), 8.0),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 9),
        ("LINEBELOW", (0,-1), (-1,-1), .6, colors.HexColor("#dbe5ec")),
    ]
    if header:
        estilos.extend([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3348")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "FuenteEDAB"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5f9")]),
        ])
    else:
        estilos.append(("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f0f5f9")]))
    t.setStyle(TableStyle(estilos))
    return t


def main():
    archivo = OUT / "resumen_lora.json"
    if not archivo.is_file():
        raise FileNotFoundError("Primero ejecutar entrenar_lora.py con GPU y generar resultados reales")
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    validar(datos)
    regular, negrita = fuente()
    REPORTES.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(DESTINO), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.9 * cm, bottomMargin=1.7 * cm,
        title="Fine-tuning eficiente con LoRA | AG News",
        author="Data Science III",
    )
    anchura = A4[0] - 3.8*cm
    azul = colors.HexColor("#20364a")
    teal = colors.HexColor("#007f8c")
    p = getSampleStyleSheet()
    p.add(ParagraphStyle(name="TituloEDD", fontName=negrita, fontSize=20, leading=26,
                         textColor=azul, spaceAfter=10))
    p.add(ParagraphStyle(name="SubEDD", fontName=negrita, fontSize=12, leading=16,
                         textColor=teal, spaceBefore=13, spaceAfter=6))
    p.add(ParagraphStyle(name="TextoEDD", fontName=regular, fontSize=9, leading=14.5,
                         textColor=azul, spaceAfter=8))
    p.add(ParagraphStyle(name="NotaEDD", fontName=regular, fontSize=7.8, leading=12,
                         textColor=colors.HexColor("#607489"), spaceAfter=8))

    def T(s):return Paragraph(s, p["TextoEDD"])
    def H(s):return Paragraph(s, p["SubEDD"])
    def nota(s):return Paragraph(s, p["NotaEDD"])
    def grafico(nombre, ancho=15.5*cm, alto=8.8*cm):
        return Image(str(OUT/nombre), width=ancho, height=alto, kind="proportional")
    def pie(canvas, documento):
        canvas.setStrokeColor(colors.HexColor("#cedce4"))
        canvas.line(1.9*cm, 1.40*cm, A4[0]-1.9*cm, 1.40*cm)
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#607489"))
        canvas.drawString(1.9*cm, 1.12*cm, "Data Science III · Checkpoint 4 · AG News")
        canvas.drawRightString(A4[0]-1.9*cm, 1.12*cm, f"Página {documento.page}")

    base = datos["baseline_checkpoint3"]
    lora = datos["lora"]
    param = lora["parametros"]
    tr = datos["entrenamiento"]
    met = datos["test"]
    detalle = pd.read_csv(OUT / "classification_report_lora.csv", index_col=0)
    matriz_csv = pd.read_csv(OUT / "matriz_confusion_lora.csv", index_col=0)
    if list(matriz_csv.index) != datos["etiquetas"] or list(matriz_csv.columns) != datos["etiquetas"]:
        raise ValueError("El orden de etiquetas no coincide con la matriz de confusión")
    if int(matriz_csv.to_numpy().sum()) != 2000:
        raise ValueError("La matriz de confusión debe representar 2.000 noticias")
    errores = []
    for i, origen in enumerate(datos["etiquetas"]):
        for j, destino in enumerate(datos["etiquetas"]):
            if i != j and int(matriz_csv.iloc[i, j]) > 0:
                errores.append((int(matriz_csv.iloc[i, j]), origen, destino))
    errores.sort(reverse=True)
    principal_confusion = (f"La confusión más frecuente fue {errores[0][1]} como {errores[0][2]} "
                           f"({errores[0][0]} noticias)." if errores else
                           "No se registraron errores entre categorías en esta ejecución.")
    config = datos["hiperparametros"]
    gpu = datos["hardware"]
    epoca = config["epoca_mejor"]
    if tr["perdida_train_en_mejor_epoca"] is None:
        raise ValueError("Falta registrar la pérdida de entrenamiento correspondiente a la mejor época")

    historia = [
        Paragraph("DATA SCIENCE III | NLP &amp; DEEP LEARNING", p["NotaEDD"]),
        Paragraph("Clasificación de noticias con LoRA", p["TituloEDD"]),
        T("<b>Pre-entrega 4. Fine-tuning eficiente de un Transformer sobre AG News.</b> "
          "Se compara el modelo neuronal con el clasificador TF-IDF del módulo anterior, "
          "conservando la misma muestra académica y el mismo conjunto oficial de prueba."),
        H("1. Arquitectura y elección del modelo"),
        T("Se utiliza <b>DistilBERT base uncased</b>, un Transformer preentrenado en inglés. "
          "Se eligió porque las noticias del corpus están en inglés y su tamaño permite ajustar adaptadores "
          "en una GPU gratuita. El modelo produce cuatro etiquetas: Business, Sci_Tech, Sports y World. "
          "La elección se apoya en Sanh et al. (2019), sin asumir que sea superior al baseline [1]."),
        tabla([
            ["Característica", "Configuración aplicada"],
            ["Arquitectura", datos["modelo"]],
            ["Datos", "AG News · 8.000 entrenamiento · 2.000 prueba"],
            ["Tokenización", "WordPiece, padding dinámico, truncamiento"],
            ["Longitud máxima", f"{datos['procesamiento']['max_length']} sub-tokens"],
            ["Hardware", f"{gpu['gpu']} ({gpu['vram_gb']} GiB de VRAM)"],
        ], [anchura*.33, anchura*.67]),
        H("2. Configuración PEFT-LoRA"),
        T("Se inyectan adaptadores en las proyecciones de <b>query</b> y <b>value</b> "
          "de los seis bloques de atención. Las matrices preentrenadas quedan congeladas. "
          "Las capas <i>pre_classifier</i> y <i>classifier</i> sí se entrenan y guardan, "
          "porque deben aprender las cuatro categorías de este experimento [2, 3]."),
        tabla([
            ["Parámetro", "Valor registrado"],
            ["Rango y escala", f"r={lora['r']}; alpha={lora['alpha']}; dropout={lora['lora_dropout']}"],
            ["Módulos objetivo", ", ".join(lora["target_modules"])],
            ["Parámetros del modelo original", numero(param["modelo_original"])],
            ["Parámetros totales con LoRA", numero(param["modelo_con_adaptadores"])],
            ["Parámetros entrenables", numero(param["entrenables"])],
            ["Porcentaje entrenable", porcentaje(param["fraccion_entrenable"])],
            ["Solo matrices LoRA", numero(param["solo_lora"])],
            ["Cabeza y otros módulos entrenables", numero(param["cabeza_y_otros_entrenables"])],
        ], [anchura*.53, anchura*.47]),
        nota("El porcentaje incluye adaptadores y la cabeza clasificadora (inicializada para cuatro clases). "
             "No equivale a porcentaje de VRAM: activaciones y estados del optimizador también consumen recursos."),
        PageBreak(),
        H("3. Entrenamiento, validación y eficiencia"),
        T("Se separaron estratificadamente 6.400 noticias para ajuste y 1.600 para validación, "
          "sin textos normalizados duplicados entre particiones. Se seleccionó la época por F1-macro "
          "de validación. El archivo de prueba no participó en esa selección."),
        tabla([
            ["Indicador", "Valor medido"],
            ["Tasa de aprendizaje", f"{config['learning_rate']:g}"],
            ["Batch y acumulación", f"{config['batch_size']} por dispositivo; x{config['acumulacion_gradientes']}"],
            ["Épocas máximas / seleccionada", f"{config['epocas_maximas']} / {epoca}"],
            ["Loss de entrenamiento (mejor época)", f"{tr['perdida_train_en_mejor_epoca']:.4f}"],
            ["Loss de validación (mejor época)", f"{tr['perdida_validacion_mejor_epoca']:.4f}"],
            *([ ["Loss de reajuste final (8.000)", f"{tr['perdida_entrenamiento_final_8k']:.4f}"] ]
              if tr.get("perdida_entrenamiento_final_8k") is not None else []),
            ["F1-macro de validación", porcentaje(tr["f1_macro_validacion_mejor"])],
            ["Tiempo de selección", f"{tr['tiempo_seleccion_seg']/60:.1f} minutos"],
            ["Tiempo de ajuste final", f"{tr['tiempo_reajuste_seg']/60:.1f} minutos"],
            ["Tiempo total de entrenamiento", f"{tr['tiempo_total_entrenamiento_seg']/60:.1f} minutos"],
            ["Pico de memoria PyTorch", f"{gpu.get('pico_memoria_asignada_gb', float('nan')):.2f} GiB asignados"],
        ], [anchura*.58, anchura*.42]),
        Spacer(1, 10),
        grafico("curvas_loss.png", ancho=anchura, alto=8.0*cm),
        nota("Las curvas muestran la fase de selección por validación. "
             + ("La evaluación de test utiliza un nuevo ajuste sobre las 8.000 noticias, durante "
                "el número de épocas seleccionado." if tr["reajuste_completo"] else
                "El modelo evaluado en test es el elegido con la fase de selección (6.400 noticias).")),
        PageBreak(),
        H("4. Evaluación y comparación con el baseline"),
        T("Los resultados siguientes corresponden a las mismas 2.000 noticias oficiales de test. "
          "El vectorizador TF-IDF del Checkpoint 3 nunca se ajustó sobre esas noticias."),
        tabla([
            ["Métrica", "TF-IDF + Naive Bayes", "DistilBERT + LoRA"],
            ["Precision macro", porcentaje(base["precision_macro"]), porcentaje(met["precision_macro"])],
            ["Recall macro", porcentaje(base["recall_macro"]), porcentaje(met["recall_macro"])],
            ["F1-macro", porcentaje(base["f1_macro"]), porcentaje(met["f1_macro"])],
            ["Accuracy", porcentaje(base["accuracy"]), porcentaje(met["accuracy"])],
        ], [anchura*.34, anchura*.33, anchura*.33]),
        Spacer(1, 12),
        grafico("matriz_confusion_lora.png", ancho=13.6*cm, alto=11.0*cm),
        nota("Las cifras de TF-IDF se leen directamente del reporte registrado en el Checkpoint 3. "
             "Las cifras de LoRA y la matriz de confusión proceden de la ejecución actual."),
        H("Desempeño por categoría"),
        tabla([["Categoría", "Precision", "Recall", "F1"]] + [
            [c, porcentaje(float(detalle.loc[c, "precision"])),
                porcentaje(float(detalle.loc[c, "recall"])),
                porcentaje(float(detalle.loc[c, "f1-score"]))]
            for c in datos["etiquetas"]
        ], [anchura*.35, anchura*.21, anchura*.21, anchura*.23]),
        T(principal_confusion),
        PageBreak(),
        H("5. Interpretación y conclusiones técnicas"),
    ]

    delta = (met["f1_macro"]-base["f1_macro"])*100
    if delta > .5:
        lectura = (f"LoRA aumentó el F1-macro en {delta:.2f} puntos porcentuales. "
                   "Se observa una mejora en esta muestra, aunque su conveniencia operativa depende del tiempo "
                   "y de la memoria exigida por el entrenamiento.")
    elif delta >= -.5:
        lectura = (f"La diferencia de F1-macro respecto de TF-IDF fue de {delta:+.2f} puntos porcentuales. "
                   "Los modelos ofrecen resultados próximos, por lo que el clasificador clásico "
                   "sigue siendo una alternativa de menor complejidad.")
    else:
        lectura = (f"LoRA obtuvo un F1-macro inferior al baseline por {abs(delta):.2f} puntos porcentuales. "
                   "En esta ejecución el mayor costo computacional no produjo una mejora de calidad.")

    historia.extend([
        T(lectura),
        T("Se preservó la separación entre entrenamiento, validación y prueba. La pérdida de entrenamiento "
          "y la de validación permiten evaluar la convergencia; la matriz de confusión permite revisar "
          "qué categorías presentan errores sistemáticos. La comparación por F1-macro es apropiada porque "
          "las cuatro categorías están equilibradas [5]. Diferencias pequeñas no prueban "
          "por sí solas superioridad estadística."),
        T(f"El ajuste con LoRA requirió <b>{tr['tiempo_total_entrenamiento_seg']/60:.1f} minutos</b> "
          f"en <b>{gpu['gpu']}</b>. Sin una medición del baseline clásico en el mismo hardware, "
          "no sería riguroso informar una aceleración cuantitativa. Asimismo, la reducción de parámetros "
          "entrenables no significa que el consumo total de memoria disminuya en la misma proporción."),
        nota("Alcance: este experimento se limita a la muestra académica de AG News. "
             "El test ya fue evaluado en checkpoints anteriores y no representa un conjunto externo nunca consultado. "
             "No se ensayó un fine-tuning completo como control, por lo que no se infiere una ventaja causal de LoRA "
             "frente a actualizar todos los pesos. Las métricas pueden variar con semillas, bibliotecas y GPU."),
        H("6. Datos y reproducibilidad"),
        tabla([
            ["Componente", "Versión / procedencia"],
            ["Dataset", "CSV originales de AG News facilitados por la cátedra"],
            ["PyTorch", datos["versiones"]["torch"]],
            ["Transformers", datos["versiones"]["transformers"]],
            ["PEFT", datos["versiones"]["peft"]],
            ["Datasets", datos["versiones"]["datasets"]],
            ["Semilla", str(config["semilla"])],
            ["Test", "Sin ajustar LoRA ni seleccionar hiperparámetros con él"],
        ], [anchura*.32, anchura*.68]),
        nota("Las versiones, los parámetros y los identificadores SHA-256 de los archivos CSV "
             "se registran en outputs/resumen_lora.json."),
        H("7. Referencias bibliográficas y técnicas"),
        nota("[1] Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. "
             "https://arxiv.org/abs/1910.01108"),
        nota("[2] Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. "
             "https://arxiv.org/abs/2106.09685"),
        nota("[3] Hugging Face. Documentación PEFT y LoRA. "
             "https://huggingface.co/docs/peft/package_reference/lora"),
        nota("[4] Hugging Face. Trainer / TrainingArguments. "
             "https://huggingface.co/docs/transformers/main_classes/trainer"),
        nota("[5] Scikit-learn. Métrica F1 para clasificación multiclase. "
             "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"),
    ])
    doc.build(historia, onFirstPage=pie, onLaterPages=pie)
    print(f"PDF generado: {DESTINO} ({DESTINO.stat().st_size/1024:.1f} KB)")
    return DESTINO


if __name__ == "__main__":
    main()
