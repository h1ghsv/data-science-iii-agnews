# Checkpoint 4 | Fine-tuning con LoRA

## Objetivo

Se implementa un clasificador de noticias con **DistilBERT** y adaptadores **LoRA** para las cuatro categorías del corpus AG News. Los resultados se comparan con el modelo TF-IDF + Naive Bayes del Checkpoint 3.

## Dataset

Se utilizan los archivos `data/ag_news_train.csv` (8.000 noticias) y `data/ag_news_test.csv` (2.000 noticias), provistos por la cátedra. Las etiquetas se codifican como enteros: Business (0), Sci_Tech (1), Sports (2) y World (3).

Para elegir la cantidad de épocas se utiliza una división estratificada de 6.400 textos de entrenamiento y 1.600 de validación. Se comprueba que no haya textos normalizados duplicados entre esas particiones. El conjunto de prueba se utiliza para la evaluación final, sin intervenir en la elección de hiperparámetros.

## Modelo y preprocesamiento

Se eligió `distilbert-base-uncased` porque el corpus está en inglés y este modelo es más liviano que BERT base, por lo que resulta adecuado para trabajar con una GPU de Google Colab. Se usa su tokenizador WordPiece, truncamiento a 128 sub-tokens y padding dinámico por lote.

En esta etapa se eliminan etiquetas HTML, enlaces y espacios repetidos, pero se mantiene el texto necesario para interpretar el contexto. No se eliminan stopwords ni se lematizan las palabras antes del tokenizador del Transformer.

## Configuración de LoRA

| Parámetro | Valor |
|---|---|
| `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_lin`, `v_lin` |
| Tarea | Clasificación de secuencias, 4 clases |
| Cabeza de clasificación | `pre_classifier`, `classifier` |
| Tasa de aprendizaje | 1e-4 |
| Épocas máximas | 3 |
| Batch por dispositivo | 16 |
| Acumulación de gradientes | 2 |
| Semilla | 42 |

LoRA modifica solo una parte de las matrices de atención. El rango `r=8` permite incorporar adaptadores pequeños y `alpha=16` determina su escala. También se entrenan las capas de clasificación para adaptar la salida a las cuatro categorías de AG News.

El número de parámetros totales, los parámetros entrenables y su porcentaje se calculan durante la ejecución. Se registran el hardware utilizado, las pérdidas y el tiempo de entrenamiento para evaluar el costo de este enfoque.

## Entrenamiento y evaluación

Los modelos se comparan mediante **Precision macro, Recall macro y F1-macro**, además de Accuracy, métricas por categoría y matriz de confusión. Se utiliza F1-macro porque las cuatro clases tienen la misma cantidad de documentos.

La mejor época se selecciona mediante validación. La ejecución con `--refit-full` vuelve a iniciar el modelo y lo entrena con las 8.000 noticias durante la cantidad de épocas seleccionada, antes de evaluar el conjunto de prueba. Esto permite compararlo con el baseline del Checkpoint 3, que también se entrenó con las 8.000 noticias.

El baseline obtuvo en test: **Precision macro 89,14 %, Recall macro 89,10 % y F1-macro 89,11 %**. Estos valores se leen de los archivos de resultados del Checkpoint 3. Los valores de LoRA, el tiempo y los parámetros entrenables se obtienen al ejecutar el experimento en GPU.

## Ejecución en Google Colab

Seleccionar una GPU T4 y abrir el notebook [`Fine_Tuning_LoRA_AGNews_Colab.ipynb`](../notebooks/Fine_Tuning_LoRA_AGNews_Colab.ipynb). También es posible ejecutar desde la raíz del repositorio:

```bash
python -m pip install -r checkpoint_4_lora/requirements_lora.txt
python -m pytest -q checkpoint_4_lora/tests
python -m checkpoint_4_lora.entrenar_lora --epochs 3 --max-length 128 --refit-full
python -m checkpoint_4_lora.generar_pdf
```

La ejecución requiere PyTorch con CUDA y conexión para descargar DistilBERT. El archivo principal a entregar es `checkpoint_4_lora/reportes/Arevalo_JuanPablo_Checkpoint_NLP3.pdf`, que incluye la arquitectura, los parámetros de LoRA, las pérdidas, las métricas, la comparación y las conclusiones.

## Alcance

El uso de LoRA reduce la cantidad de parámetros entrenados, aunque no implica necesariamente una reducción equivalente del tiempo o de la memoria de GPU. La comparación permite estudiar si la diferencia de F1-macro compensa el mayor costo computacional frente a TF-IDF. El conjunto de prueba también fue utilizado en los checkpoints anteriores, por lo que no constituye una evaluación externa nueva para todo el proyecto.

## Referencias

1. Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. https://arxiv.org/abs/1910.01108
2. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685
3. Hugging Face. *LoRA configuration*. https://huggingface.co/docs/peft/package_reference/lora
4. Hugging Face. *Trainer*. https://huggingface.co/docs/transformers/main_classes/trainer
