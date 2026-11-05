"""Normalización y tokenización de AG News, coherentes con el checkpoint 2."""
from __future__ import annotations

import html
import re
from collections.abc import Iterable

import spacy


NEGACIONES = {"no", "not", "nor"}


def limpiar_texto(texto: str) -> str:
    texto = str(texto)
    texto = re.sub(r"(?<!&)#(\d{2,5});", r"&#\1;", texto)
    texto = re.sub(r"(?<!&)\b(quot|amp|nbsp|lt|gt);", r"&\1;", texto, flags=re.I)
    texto = html.unescape(html.unescape(texto))
    for patron, reemplazo in [
        (r"\bU\.S\.", "USA"),
        (r"\bU\.K\.", "UK"),
        (r"\bE\.U\.", "EU"),
    ]:
        texto = re.sub(patron, reemplazo, texto, flags=re.I)
    texto = re.sub(r"<[^>]*>", " ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+|\b[\w.%-]+@[\w.-]+\.[a-z]{2,}\b", " ", texto, flags=re.I)
    texto = re.sub(r"\b[a-z0-9-]+\.(?:com|org|net|gov|edu)\b", " ", texto, flags=re.I)
    texto = re.sub(r"\(\s*(?:ap|reuters|afp)\s*\)", " ", texto, flags=re.I)
    texto = re.sub(r"\b(?:reuters|associated press|canadian press)\s*[-:]", " ", texto, flags=re.I)
    texto = re.sub(r"\bquote\s*[,;:]\s*profile\s*[,;:]\s*research\b", " ", texto, flags=re.I)
    texto = re.sub(r"\bquote\s+profile\s+research\b", " ", texto, flags=re.I)
    texto = texto.lower().replace("’", "'")
    texto = re.sub(r"\bcan't\b", "can not", texto)
    texto = re.sub(r"\bwon't\b", "will not", texto)
    texto = re.sub(r"n\s*'t\b", " not", texto)
    texto = re.sub(r"\b([a-z]+)'s\b", r"\1", texto)
    texto = re.sub(r"[^a-z\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def crear_nlp(lematizar: bool = False):
    if lematizar:
        try:
            return spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                "Falta el modelo en_core_web_sm. Ejecutá: python -m spacy download en_core_web_sm"
            ) from exc
    return spacy.blank("en")


def preparar_documentos(textos: Iterable[str], *, lematizar: bool = False) -> list[str]:
    """Tokeniza con SpaCy y, opcionalmente, lematiza con su modelo inglés."""
    nlp = crear_nlp(lematizar)
    limpios = (limpiar_texto(t) for t in textos)
    preparados = []
    for doc in nlp.pipe(limpios, batch_size=128, n_process=1):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            palabra = (token.lemma_ if lematizar else token.text).lower().strip()
            if palabra.isalpha() and len(palabra) > 1:
                tokens.append(palabra)
        preparados.append(" ".join(tokens))
    return preparados


def stopwords_ingles() -> list[str]:
    palabras = spacy.blank("en").Defaults.stop_words
    return sorted((set(palabras) - NEGACIONES) | {"ll", "ve"})
