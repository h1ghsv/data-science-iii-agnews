# Checkpoint 2 | Análisis exploratorio de AG News

En esta etapa se estudia el corpus de noticias de la cátedra antes de los modelos de NLP. Se utilizan exclusivamente los **8.000 textos del archivo de entrenamiento** para calcular frecuencias, longitudes y vocabulario. Las **2.000 noticias de test** quedan reservadas para la evaluación posterior.

## Contenido

El informe analiza la calidad y la limpieza del texto, distribución de clases, longitudes globales y por categoría, percentiles y cobertura de `max_len`, las 50 palabras más frecuentes, stopwords, 20 bigramas y 20 trigramas, Ley de Zipf, riqueza léxica, vocabulario distintivo y similitud léxica entre categorías. El informe final se genera con tokenización y lematización de SpaCy (`en_core_web_sm`).

## Ejecutar desde la raíz del repositorio

```bash
python -m pip install -r checkpoint_2_eda/requirements_eda.txt
python -m spacy download en_core_web_sm
python checkpoint_2_eda/generar_reporte.py
```

Los datos deben estar ubicados en:

- `data/ag_news/ag_news_train.csv`
- `data/ag_news/ag_news_test.csv`

Los resultados se escriben en `checkpoint_2_eda/outputs/` y el documento final en `checkpoint_2_eda/reportes/EDA_NLP_ArevaloJuanPablo.pdf`.

En Linux o Google Colab, si falta DejaVu Sans, se puede instalar `fonts-dejavu-core`. La fuente es una dependencia de presentación, no del procesamiento del corpus.

## Metodología

La función de normalización corrige entidades HTML, suprime etiquetas, enlaces y residuos editoriales, y regulariza los espacios. Se aplica SpaCy para tokenizar y obtener los lemas. Se conserva un conjunto con stopwords para el diagnóstico de frecuencias y otro sin ellas para calcular N-gramas, preservando negaciones relevantes. La elección futura de `max_len` se fundamenta en percentiles de longitud; en Transformers debe volverse a calcular mediante el tokenizador subword correspondiente.

## Alcance

El análisis exploratorio no mide el rendimiento de un clasificador ni reemplaza la evaluación del checkpoint 1. Sus resultados corresponden al corpus provisto por la cátedra; el análisis no utiliza el test para tomar decisiones sobre vocabulario o hiperparámetros. Los CSV son material académico y su publicación en un repositorio público debe respetar los permisos de la cátedra.
