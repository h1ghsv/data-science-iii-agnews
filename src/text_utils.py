"""Normalización textual compartida por entrenamiento e inferencia."""
import html
import re


def preprocess_text(text: str) -> str:
    """Normaliza el texto conservando negaciones y números."""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'(?:https?://|www\.)\S+', ' ', text, flags=re.I)
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

