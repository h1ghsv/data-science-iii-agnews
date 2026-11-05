"""Pruebas de las operaciones críticas del clasificador."""
import numpy as np

from checkpoint_3_tfidf.entrenar import crear_modelo, crear_vectorizador, encontrar_dato
from checkpoint_3_tfidf.preprocesamiento import limpiar_texto, preparar_documentos, stopwords_ingles


def test_ubicacion_csv():
    assert encontrar_dato("ag_news_train.csv").is_file()
    assert encontrar_dato("ag_news_test.csv").is_file()


def test_limpieza_html_y_negacion():
    texto = limpiar_texto("This is not great &#39;news&#39;! http://example.com")
    assert "not" in texto.split()
    assert "news" in texto.split()
    assert "http" not in texto.split()


def test_tokenizador_spacy():
    salida = preparar_documentos(["The researchers are testing new batteries."])
    assert len(salida) == 1
    assert "researchers" in salida[0]
    assert "batteries" in salida[0]


def test_vectorizador_solo_fit_train():
    config = {"max_features": 100, "ngram_range": (1, 2)}
    vec = crear_vectorizador(config)
    train = ["computer software linux server", "sport team championship win"] * 3
    validacion = ["unseenneologism computer technology"]
    train_matriz = vec.fit_transform(train)
    val_matriz = vec.transform(validacion)
    assert "unseenneologism" not in vec.vocabulary_
    assert train_matriz.shape[1] == val_matriz.shape[1]
    assert "not" not in stopwords_ingles()


def test_clasificadores_disponibles():
    for modelo in ("NB", "LR", "SVM"):
        assert hasattr(crear_modelo(modelo), "fit")


def test_sin_duplicados_entre_train_y_validacion():
    from checkpoint_3_tfidf.entrenar import evitar_duplicados_en_validacion
    docs = ["same", "same", "another", "unique", "hello", "world"]
    labels = ["A", "A", "A", "A", "B", "B"]
    train, val = evitar_duplicados_en_validacion([0, 2, 4], [1, 3, 5], docs, labels)
    assert len(train) == 3 and len(val) == 3
    assert not {docs[i] for i in train} & {docs[i] for i in val}
