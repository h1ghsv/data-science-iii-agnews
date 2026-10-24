"""Predicción temática de noticias con el modelo entrenado."""
import argparse
from pathlib import Path
import joblib
import torch
from .train import NewsMLP, ROOT, select_device


def predict_texts(texts):
    vec = joblib.load(ROOT / 'models/tfidf_train_only.joblib')
    try:
        checkpoint = torch.load(ROOT / 'models/mejor_modelo.pt', map_location='cpu', weights_only=True)
    except FileNotFoundError as e:
        raise RuntimeError('Ejecutá primero python -m src.train') from e
    device = select_device()
    model = NewsMLP(checkpoint['vocab_size'], checkpoint['hidden_size'],
                    len(checkpoint['class_names']), checkpoint['dropout']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        X = torch.from_numpy(vec.transform(texts).toarray().astype('float32')).to(device)
        probabilities = torch.softmax(model(X), dim=1).cpu().numpy()
    return [(checkpoint['class_names'][int(row.argmax())], round(float(row.max()), 4))
            for row in probabilities]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('text', help='Texto de noticia a clasificar (inglés)')
    a = p.parse_args()
    print(predict_texts([a.text])[0])
