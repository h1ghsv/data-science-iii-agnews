"""Fine-tuning PEFT-LoRA de DistilBERT sobre AG News. Ejecutar desde la raíz del repositorio."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODEL_DIR = BASE / "models"
LABELS = ("Business", "Sci_Tech", "Sports", "World")
SEED = 42
MODEL_ID = "distilbert-base-uncased"


def encontrar_csv(nombre: str) -> Path:
    for carpeta in (ROOT / "data", ROOT / "data" / "ag_news"):
        candidato = carpeta / nombre
        if candidato.is_file():
            return candidato
    raise FileNotFoundError(f"No se encontró {nombre} en data/ ni en data/ag_news/")


def normalizar_texto(texto: str) -> str:
    """Limpieza estructural ligera: conserva contexto para WordPiece y negaciones."""
    texto = html.unescape(str(texto))
    texto = re.sub(r"<[^>]*>", " ", texto)
    texto = re.sub(r"(?:https?://|www\.)\S+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def clave_duplicado(texto: str) -> str:
    return re.sub(r"\W+", " ", normalizar_texto(texto).lower()).strip()


def dividir_sin_duplicados(textos: list[str], etiquetas: list[int]):
    """Partición estratificada sin textos normalizados repetidos entre train y val."""
    indices = np.arange(len(textos))
    ids_train, ids_val = train_test_split(
        indices, test_size=0.20, stratify=etiquetas, random_state=SEED
    )
    conjunto_train = set(map(int, ids_train))
    conjunto_val = set(map(int, ids_val))
    claves = [clave_duplicado(t) for t in textos]
    conteos = {}
    for clave in claves:
        conteos[clave] = conteos.get(clave, 0) + 1
    while True:
        train_keys = {claves[i] for i in conjunto_train}
        duplicado = next((i for i in sorted(conjunto_val) if claves[i] in train_keys), None)
        if duplicado is None:
            break
        val_keys = {claves[i] for i in conjunto_val}
        reemplazo = next((i for i in sorted(conjunto_train)
                          if etiquetas[i] == etiquetas[duplicado]
                          and claves[i] not in val_keys
                          and conteos[claves[i]] == 1), None)
        if reemplazo is None:
            raise RuntimeError("No es posible separar los duplicados conservando balance")
        conjunto_val.remove(duplicado)
        conjunto_train.add(duplicado)
        conjunto_train.remove(reemplazo)
        conjunto_val.add(reemplazo)
    assert len(conjunto_train) == 6400 and len(conjunto_val) == 1600
    assert not {claves[i] for i in conjunto_train} & {claves[i] for i in conjunto_val}
    return np.array(sorted(conjunto_train)), np.array(sorted(conjunto_val))


def cargar_corpus():
    ruta_train = encontrar_csv("ag_news_train.csv")
    ruta_test = encontrar_csv("ag_news_test.csv")
    train = pd.read_csv(ruta_train)
    test = pd.read_csv(ruta_test)
    for nombre, df in [("train", train), ("test", test)]:
        if list(df.columns) != ["text", "label"]:
            raise ValueError(f"{nombre}: se esperan columnas text y label")
        if df.isna().any().any():
            raise ValueError(f"{nombre}: contiene valores nulos")
        if not set(df.label).issubset(LABELS):
            raise ValueError(f"{nombre}: contiene etiquetas inesperadas")
    if len(train) != 8000 or len(test) != 2000:
        raise ValueError("El dataset no coincide con la muestra académica 8.000/2.000")
    train = train.copy()
    test = test.copy()
    train["text"] = train["text"].map(normalizar_texto)
    test["text"] = test["text"].map(normalizar_texto)
    mapa = {nombre: i for i, nombre in enumerate(LABELS)}
    train["labels"] = train.label.map(mapa).astype(int)
    test["labels"] = test.label.map(mapa).astype(int)
    if set(map(clave_duplicado, train.text)) & set(map(clave_duplicado, test.text)):
        raise ValueError("Se encontraron textos coincidentes entre train y test")
    return train, test, ruta_train, ruta_test


def metrica_macro(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def baseline_guardado() -> dict:
    archivo = ROOT / "checkpoint_3_tfidf" / "outputs" / "reporte_clasificacion.csv"
    if not archivo.is_file():
        raise FileNotFoundError("Se necesita el reporte real de clasificación del Checkpoint 3")
    tabla = pd.read_csv(archivo, index_col=0, encoding="utf-8-sig")
    fila = tabla.loc["macro avg"]
    return {
        "modelo": "TF-IDF + Multinomial Naive Bayes (Checkpoint 3)",
        "precision_macro": float(fila["precision"]),
        "recall_macro": float(fila["recall"]),
        "f1_macro": float(fila["f1-score"]),
        "accuracy": float(tabla.loc["accuracy", "precision"]),
        "entrenamiento_documentos": 8000,
        "archivo": str(archivo.relative_to(ROOT)),
    }


def dibujar_confusion(matriz: np.ndarray, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 5.8))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(4), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Categoría predicha")
    ax.set_ylabel("Categoría real")
    ax.set_title("DistilBERT + LoRA | Matriz de confusión")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > matriz.max() * .55 else "black")
    fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def dibujar_perdidas(logs: list[dict], destino: Path) -> None:
    entrenamiento = [(float(x["epoch"]), float(x["loss"])) for x in logs
                     if "loss" in x and "epoch" in x]
    validacion = [(float(x["epoch"]), float(x["eval_loss"])) for x in logs
                  if "eval_loss" in x and "epoch" in x]
    fig, ax = plt.subplots(figsize=(8, 4.1))
    if entrenamiento:
        ax.plot(*zip(*entrenamiento), marker="o", label="Entrenamiento")
    if validacion:
        ax.plot(*zip(*validacion), marker="s", label="Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Evolución de pérdidas durante selección del modelo")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def convertir_nativo(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--refit-full", action="store_true",
                        help="Vuelve a entrenar sobre las 8.000 noticias tras seleccionar época en validación")
    args = parser.parse_args()
    if args.epochs < 1 or args.r < 1 or args.max_length < 16:
        parser.error("Verificá epochs, r y max-length")

    import torch
    import transformers
    import datasets
    import peft
    import accelerate
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita GPU CUDA. En Colab: Entorno de ejecución > Cambiar tipo > T4 GPU")
    gpu = torch.cuda.get_device_name(0)
    print("Dispositivo:", gpu, flush=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    OUT.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    torch.cuda.reset_peak_memory_stats(0)

    train_df, test_df, ruta_train, ruta_test = cargar_corpus()
    indices_train, indices_val = dividir_sin_duplicados(
        train_df.text.tolist(), train_df.labels.tolist()
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def como_dataset(tabla: pd.DataFrame):
        datos = Dataset.from_dict({
            "text": tabla.text.tolist(),
            "labels": [int(x) for x in tabla.labels.tolist()],
        })
        return datos.map(lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=args.max_length
        ), batched=True, remove_columns=["text"])

    train_ds = como_dataset(train_df.iloc[indices_train])
    val_ds = como_dataset(train_df.iloc[indices_val])
    test_ds = como_dataset(test_df)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    id2label = dict(enumerate(LABELS))
    label2id = {v:k for k,v in id2label.items()}

    def construir_modelo():
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=4, id2label=id2label, label2id=label2id
        )
        objetivos = [n for n, _ in base.named_modules()
                     if n.endswith("q_lin") or n.endswith("v_lin")]
        if len(objetivos) != 12:
            raise RuntimeError(f"Módulos esperados de atención: 12; encontrados {len(objetivos)}")
        base_parametros = sum(p.numel() for p in base.parameters())
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
            bias="none",
        )
        modelo = get_peft_model(base, config)
        total = sum(p.numel() for p in modelo.parameters())
        entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        lora_puros = sum(p.numel() for n,p in modelo.named_parameters()
                         if p.requires_grad and "lora_" in n)
        if entrenables == 0 or entrenables / total >= .03:
            raise RuntimeError(f"Fracción de parámetros inesperada: {entrenables/total:.2%}")
        return modelo, {
            "modelo_original": base_parametros,
            "modelo_con_adaptadores": total,
            "entrenables": entrenables,
            "solo_lora": lora_puros,
            "cabeza_y_otros_entrenables": entrenables - lora_puros,
            "fraccion_entrenable": entrenables/total,
        }

    def argumentos(carpeta: Path, n_epochs: int, validacion: bool):
        return TrainingArguments(
            output_dir=str(carpeta),
            num_train_epochs=n_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            eval_strategy="epoch" if validacion else "no",
            save_strategy="epoch" if validacion else "no",
            load_best_model_at_end=validacion,
            metric_for_best_model="f1_macro" if validacion else None,
            greater_is_better=True if validacion else None,
            save_total_limit=1,
            logging_strategy="epoch",
            warmup_ratio=.1,
            weight_decay=.01,
            fp16=True,
            seed=SEED, data_seed=SEED,
            report_to="none",
            optim="adamw_torch",
            dataloader_num_workers=0,
        )

    def calcular_metricas(p):
        pred = p.predictions.argmax(axis=-1)
        y = p.label_ids
        resultado = metrica_macro(y, pred)
        return {"accuracy": resultado["accuracy"], "precision_macro": resultado["precision_macro"],
                "recall_macro": resultado["recall_macro"], "f1_macro": resultado["f1_macro"]}

    print("Cargando arquitectura DistilBERT y adaptadores LoRA...", flush=True)
    modelo, params = construir_modelo()
    print(f"Parámetros entrenables: {params['entrenables']:,} / {params['modelo_con_adaptadores']:,} "
          f"({params['fraccion_entrenable']:.2%})", flush=True)
    entrenador = Trainer(
        model=modelo, args=argumentos(MODEL_DIR / "seleccion", args.epochs, True),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tokenizer,
        compute_metrics=calcular_metricas,
    )
    inicio = time.perf_counter()
    entrenamiento = entrenador.train()
    tiempo_seleccion = time.perf_counter() - inicio
    val_metricas = entrenador.evaluate(val_ds)
    logs = entrenador.state.log_history
    mejores = [x for x in logs if "eval_f1_macro" in x]
    mejor = max(mejores, key=lambda x:x["eval_f1_macro"])
    epoca_mejor = int(round(float(mejor["epoch"])))
    perdidas_train_epoca = [x for x in logs if "loss" in x and int(round(float(x["epoch"]))) == epoca_mejor]
    perdida_train_epoca = float(perdidas_train_epoca[-1]["loss"]) if perdidas_train_epoca else None
    print(f"Mejor época: {epoca_mejor}; F1-macro val: {mejor['eval_f1_macro']:.4f}", flush=True)

    segundo_entrenamiento = None
    modelo_final = entrenador.model
    tiempo_final = 0.0
    estimacion = "6400_train_1600_validacion"
    if args.refit_full:
        print("Reentrenando desde el checkpoint base con las 8.000 noticias de train...", flush=True)
        import gc
        del entrenador, modelo, modelo_final
        gc.collect()
        torch.cuda.empty_cache()
        set_seed(SEED)
        modelo_final, _ = construir_modelo()
        entrenador_final = Trainer(
            model=modelo_final,
            args=argumentos(MODEL_DIR / "entrenamiento_final", epoca_mejor, False),
            train_dataset=como_dataset(train_df),
            data_collator=collator, processing_class=tokenizer,
        )
        inicio_final = time.perf_counter()
        segundo_entrenamiento = entrenador_final.train()
        tiempo_final = time.perf_counter()-inicio_final
        entrenador_pred = entrenador_final
        estimacion = "8000_train_refit_tras_validacion"
    else:
        entrenador_pred = entrenador

    print("Evaluación final sobre el test oficial, sin ajustar LoRA con sus etiquetas.", flush=True)
    inicio_test = time.perf_counter()
    resultado_test = entrenador_pred.predict(test_ds)
    tiempo_test = time.perf_counter() - inicio_test
    preds = np.asarray(resultado_test.predictions).argmax(axis=1)
    reales = test_df.labels.to_numpy()
    metrics_test = metrica_macro(reales, preds)
    reporte = classification_report(reales, preds, labels=[0,1,2,3], target_names=LABELS,
                                    output_dict=True, zero_division=0)
    matriz = confusion_matrix(reales, preds, labels=[0,1,2,3])
    dibujar_confusion(matriz, OUT / "matriz_confusion_lora.png")
    dibujar_perdidas(logs, OUT / "curvas_loss.png")
    pd.DataFrame(reporte).T.to_csv(OUT / "classification_report_lora.csv", encoding="utf-8-sig")
    pd.DataFrame(matriz, index=LABELS, columns=LABELS).to_csv(OUT / "matriz_confusion_lora.csv")
    (OUT / "historial_entrenamiento.json").write_text(json.dumps(logs, indent=2, default=convertir_nativo), encoding="utf-8")

    # PEFT conserva adaptadores y cabeza clasificadora; no requiere versionar pesos en GitHub.
    modelo_final.save_pretrained(MODEL_DIR / "adaptador_final")
    tokenizer.save_pretrained(MODEL_DIR / "adaptador_final")

    def hash_archivo(p: Path):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    resumen = {
        "ejecucion_real": True,
        "corpus": "AG News, muestra de Coderhouse",
        "modelo": MODEL_ID,
        "num_labels": 4,
        "etiquetas": list(LABELS),
        "dataset": {"train_total": 8000, "desarrollo": 6400, "validacion": 1600, "test": 2000,
                    "train_sha256": hash_archivo(ruta_train), "test_sha256": hash_archivo(ruta_test)},
        "procesamiento": {"limpieza": "HTML/URLs/espacios, sin eliminar negaciones ni lematizar",
                            "tokenizador": MODEL_ID, "max_length": args.max_length,
                            "padding": "dinámico por batch", "truncation": True},
        "lora": {"r": args.r, "alpha": args.alpha, "lora_dropout": .05,
                 "target_modules": ["q_lin", "v_lin"],
                 "modules_to_save": ["pre_classifier", "classifier"],
                 "task_type": "SEQ_CLS", "parametros": params},
        "hiperparametros": {"epocas_maximas": args.epochs, "epoca_mejor": epoca_mejor,
                             "batch_size": args.batch_size, "acumulacion_gradientes": 2,
                             "learning_rate": args.lr, "weight_decay": .01, "semilla": SEED},
        "entrenamiento": {"perdida_train_en_mejor_epoca": perdida_train_epoca,
                          "perdida_train_promedio_seleccion": float(entrenamiento.training_loss),
                          "perdida_validacion_mejor_epoca": float(mejor["eval_loss"]),
                          "f1_macro_validacion_mejor": float(mejor["eval_f1_macro"]),
                          "tiempo_seleccion_seg": tiempo_seleccion,
                          "reajuste_completo": bool(args.refit_full),
                          "tiempo_reajuste_seg": tiempo_final,
                          "perdida_entrenamiento_final_8k": (float(segundo_entrenamiento.training_loss)
                                                           if segundo_entrenamiento is not None else None),
                          "tiempo_total_entrenamiento_seg": tiempo_seleccion + tiempo_final,
                          "tiempo_test_seg": tiempo_test,
                          "metodo_evaluado": estimacion},
        "test": metrics_test,
        "baseline_checkpoint3": baseline_guardado(),
        "versiones": {"python": platform.python_version(), "torch": torch.__version__,
                      "transformers": transformers.__version__, "peft": peft.__version__,
                      "datasets": datasets.__version__, "accelerate": accelerate.__version__},
        "hardware": {"gpu": gpu, "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                     "pico_memoria_asignada_gb": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
                     "pico_memoria_reservada_gb": round(torch.cuda.max_memory_reserved(0) / 2**30, 3)},
        "limitaciones": ["Si no se utiliza --refit-full, LoRA se entrena con 6.400 noticias frente a 8.000 del baseline",
                         "No se informa tiempo comparable del baseline: falta medición en el mismo hardware",
                         "El conjunto test ya fue utilizado en checkpoints anteriores; no es evaluación externa totalmente ciega"],
    }
    (OUT / "resumen_lora.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2,
                                                default=convertir_nativo), encoding="utf-8")
    print("\n=== RESULTADOS REALES ===")
    print("Test:", json.dumps(metrics_test, indent=2))
    print("Segundos de entrenamiento:", round(tiempo_seleccion + tiempo_final, 1))
    print("Archivo de métricas:", OUT / "resumen_lora.json")


if __name__ == "__main__":
    main()
