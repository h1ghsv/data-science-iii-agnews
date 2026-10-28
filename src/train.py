"""Pipeline de clasificación AG News con PyTorch."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .text_utils import preprocess_text

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('Business', 'Sci_Tech', 'Sports', 'World')

@dataclass
class Config:
    seed: int = 42
    val_fraction: float = 0.20
    max_features: int = 6000
    epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.35
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    train_file: str = 'data/ag_news/ag_news_train.csv'
    test_file: str = 'data/ag_news/ag_news_test.csv'
    output_dir: str = 'outputs'
    model_dir: str = 'models'


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, os.cpu_count() or 2))


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def locate_csv(relative_path: str) -> Path:
    """Busca los datos en el diseño de carpetas actual o en data/ag_news/."""
    requested = ROOT / relative_path
    if requested.exists():
        return requested
    alternative = ROOT / 'data' / requested.name
    if alternative.exists() and requested.parent == ROOT / 'data' / 'ag_news':
        return alternative
    raise FileNotFoundError(f'No se encontró el dataset: {requested}')


def load_official_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(locate_csv(config.train_file))
    test = pd.read_csv(locate_csv(config.test_file))
    for name, frame in [('train', train), ('test', test)]:
        if set(frame.columns) != {'text', 'label'}:
            raise ValueError(f'{name}: se esperaban columnas text y label: {list(frame.columns)}')
        if frame[['text','label']].isna().any().any():
            raise ValueError(f'{name}: hay nulos; revisar los CSV originales')
        if not frame.label.isin(LABELS).all():
            raise ValueError(f'{name}: etiquetas desconocidas: {set(frame.label)-set(LABELS)}')
        if frame.text.duplicated().any():
            raise ValueError(f'{name}: hay duplicados exactos; analizar antes de dividir')
    overlap = set(train.text) & set(test.text)
    if overlap:
        raise ValueError(f'Fuga train/test: {len(overlap)} textos exactamente repetidos')
    return train, test


class NewsMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def loader_from_arrays(features, labels, batch_size, shuffle, seed):
    X = torch.from_numpy(features.toarray().astype(np.float32, copy=False))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    ds = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses = 0.0
    truth, predicted, probabilities = [], [], []
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses += loss.item() * len(yb)
            prob = torch.softmax(logits, dim=1)
            truth.extend(yb.cpu().numpy().tolist())
            predicted.extend(prob.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(prob.max(dim=1).values.cpu().numpy().tolist())
    return {
        'loss': losses / len(truth),
        'accuracy': accuracy_score(truth, predicted),
        'f1_macro': f1_score(truth, predicted, average='macro', zero_division=0),
        'f1_weighted': f1_score(truth, predicted, average='weighted', zero_division=0),
        'truth': truth, 'predicted': predicted, 'confidence': probabilities,
    }


def chart_learning(history: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, title in zip(axes, ['loss','accuracy','f1_macro'],
                                  ['Pérdida cross-entropy','Exactitud (Accuracy)','F1-macro']):
        ax.plot(history.epoch, history['train_' + metric], '-o', markersize=2.5,
                label='Entrenamiento')
        ax.plot(history.epoch, history['val_' + metric], '-o', markersize=2.5,
                label='Validación')
        ax.set_title(title)
        ax.set_xlabel('Época')
        ax.set_ylabel(metric)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('AG News | PyTorch MLP | Datos de la cátedra', fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


def chart_confusion(truth, predictions, path):
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(4), LABELS, rotation=30, ha='right')
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Clase real')
    ax.set_title('Matriz de confusión | Test final no visto')
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha='center', va='center',
                color='white' if value > (matrix.max() * .6) else '#192b38')
    fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return matrix.tolist()


def run_training(config: Config | None = None) -> dict:
    config = config or Config()
    configure_reproducibility(config.seed)
    device = select_device()
    out = ROOT / config.output_dir
    modeldir = ROOT / config.model_dir
    out.mkdir(exist_ok=True, parents=True)
    modeldir.mkdir(exist_ok=True, parents=True)

    full_train, test = load_official_data(config)
    i_train, i_val = train_test_split(
        np.arange(len(full_train)), test_size=config.val_fraction,
        stratify=full_train.label, random_state=config.seed)
    train = full_train.iloc[i_train].copy()
    val = full_train.iloc[i_val].copy()
    label_to_id = {name: i for i, name in enumerate(LABELS)}
    ytrain = train.label.map(label_to_id).to_numpy()
    yval = val.label.map(label_to_id).to_numpy()
    ytest = test.label.map(label_to_id).to_numpy()

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        tokenizer=None, token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1,2), min_df=2, max_df=.95,
        max_features=config.max_features, sublinear_tf=True, norm='l2',
        stop_words=None, dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train.text)
    X_val = vectorizer.transform(val.text)
    X_test = vectorizer.transform(test.text)
    vocab_size = X_train.shape[1]

    tr_loader = loader_from_arrays(X_train, ytrain, config.batch_size, True, config.seed)
    tr_eval_loader = loader_from_arrays(X_train, ytrain, config.batch_size, False, config.seed)
    val_loader = loader_from_arrays(X_val, yval, config.batch_size, False, config.seed)
    test_loader = loader_from_arrays(X_test, ytest, config.batch_size, False, config.seed)

    model = NewsMLP(vocab_size, config.hidden_size, len(LABELS), config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'PyTorch {torch.__version__} | device={device} | parámetros={parameter_count:,}')
    print(f'AG News cátedra: train={len(train)}, val={len(val)}, test reservado={len(test)}')
    print(f'TF-IDF: {vocab_size} atributos (fit exclusivamente en train)')

    history = []
    best_loss = float('inf')
    best_epoch, best_weights, epochs_no_improve = 0, None, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            loss.backward()
            optimizer.step()
        tr_metrics = evaluate(model, tr_eval_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch, **{f'train_{k}': tr_metrics[k]
                                   for k in ('loss','accuracy','f1_macro')},
               **{f'val_{k}': val_metrics[k]
                  for k in ('loss','accuracy','f1_macro','f1_weighted')}}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss {tr_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | "
              f"val acc {val_metrics['accuracy']:.4f} | "
              f"val F1macro {val_metrics['f1_macro']:.4f}", flush=True)
        if val_metrics['loss'] < best_loss - 0.0001:
            best_loss = val_metrics['loss']
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f'Early stopping: sin mejora {config.patience} épocas.')
                break

    pd.DataFrame(history).to_csv(out / 'historial_epocas.csv', index=False)
    chart_learning(pd.DataFrame(history), out / 'curvas_entrenamiento_validacion.png')
    model.load_state_dict(best_weights)
    test_metrics = evaluate(model, test_loader, criterion, device)
    conf = chart_confusion(test_metrics['truth'], test_metrics['predicted'],
                           out / 'matriz_confusion_test.png')
    report = classification_report(test_metrics['truth'], test_metrics['predicted'],
                                   labels=list(range(4)), target_names=list(LABELS),
                                   digits=4, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(out / 'reporte_clasificacion_test.csv')
    pd.DataFrame({'fila_test': np.arange(len(test)), 'label_real': test.label,
                  'label_predicha': [LABELS[i] for i in test_metrics['predicted']],
                  'confianza_modelo': test_metrics['confidence']}).to_csv(
                      out / 'predicciones_test.csv', index=False)
    joblib.dump(vectorizer, modeldir / 'tfidf_train_only.joblib')
    torch.save({'state_dict': model.state_dict(), 'vocab_size': vocab_size,
                'hidden_size': config.hidden_size, 'dropout': config.dropout,
                'class_names': LABELS}, modeldir / 'mejor_modelo.pt')
    summary = {
        'project': 'Checkpoint 1 | AG News cátedra | Deep Learning PyTorch',
        'data': 'AG News submuestra de la cátedra',
        'train_original': len(full_train), 'train_internal': len(train),
        'validation': len(val), 'test_reserved': len(test),
        'labels': list(LABELS), 'test_used_during_training': False,
        'device': str(device), 'torch_version': torch.__version__,
        'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
        'parameters': parameter_count,
        'vectorizer_fitted_on': 'train interno exclusivamente',
        'features': vocab_size,
        'baseline_majority_accuracy': float(train.label.value_counts(normalize=True).max()),
        'config': asdict(config), 'epochs_run': len(history),
        'best_epoch_by_val_loss': best_epoch,
        'best_val_loss': float(best_loss),
        'first_train_loss': float(history[0]['train_loss']),
        'last_train_loss': float(history[-1]['train_loss']),
        'first_val_loss': float(history[0]['val_loss']),
        'last_val_loss': float(history[-1]['val_loss']),
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']),
        'test_f1_weighted': float(test_metrics['f1_weighted']),
        'test_confusion_matrix': conf,
    }
    (out / 'resumen_experimento.json').write_text(json.dumps(summary, indent=2,
                                                           ensure_ascii=False), encoding='utf-8')
    print(f"MEJOR ÉPOCA: {best_epoch} | TEST accuracy={summary['test_accuracy']:.4f} "
          f"| F1macro={summary['test_f1_macro']:.4f}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-features', type=int, default=6000)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = Config(epochs=args.epochs, learning_rate=args.lr,
                    batch_size=args.batch_size, max_features=args.max_features)
    run_training(config)
