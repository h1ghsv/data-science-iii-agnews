"""Controles verificables sin GPU ni descarga de pesos del Transformer."""
from checkpoint_4_lora.entrenar_lora import (
    LABELS, baseline_guardado, cargar_corpus, clave_duplicado,
    dividir_sin_duplicados, normalizar_texto,
)


def test_limpieza_conserva_negacion():
    assert "not" in normalizar_texto("<p>Not good</p>  ").lower()
    assert "html" not in normalizar_texto("<b>Useful</b>").lower()


def test_etiquetas_y_baseline():
    assert LABELS == ("Business", "Sci_Tech", "Sports", "World")
    b = baseline_guardado()
    assert b["entrenamiento_documentos"] == 8000
    assert .0 < b["f1_macro"] < 1.0


def test_particion_sin_duplicados_y_datos_originales():
    train, test, _, _ = cargar_corpus()
    assert (len(train), len(test)) == (8000, 2000)
    assert list(train.groupby("label").size().values) == [2000] * 4
    tr, val = dividir_sin_duplicados(train.text.tolist(), train.labels.tolist())
    assert len(tr) == 6400 and len(val) == 1600
    assert not {clave_duplicado(train.text.iloc[i]) for i in tr} & {
        clave_duplicado(train.text.iloc[i]) for i in val
    }
