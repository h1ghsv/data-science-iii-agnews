# Metodología y resultados | AG News

## Datos y particiones

Se utilizan los CSV de AG News entregados por la cátedra. Los 8.000 registros de `train` se separan de forma estratificada en 6.400 para entrenamiento y 1.600 para validación. El conjunto oficial de 2.000 registros queda reservado para el test final. Las cuatro clases están balanceadas.

## Preprocesamiento

Se normalizan entidades HTML, etiquetas, URL, mayúsculas y espacios. El texto se representa mediante TF-IDF con unigramas y bigramas; el vocabulario se ajusta solo sobre el entrenamiento interno. No se eliminan negaciones de forma generalizada.

## Entrenamiento

MLP con 6.000 entradas, una capa oculta de 128 neuronas ReLU, dropout 0.35 y salida de cuatro logits. Optimizador Adam (`lr=0.001`, `weight_decay=0.0001`) y pérdida `CrossEntropyLoss`. Semilla 42, batch size 128 y selección del mejor estado por pérdida de validación, con paciencia de cinco épocas.

## Resultados

| Indicador | Valor |
|---|---:|
| PyTorch | 2.10.0+cpu |
| Época seleccionada | 8 |
| Mínimo `val_loss` | 0.3441 |
| Épocas ejecutadas | 13 |
| Accuracy en test | 0.8820 |
| F1-macro en test | 0.8821 |
| Loss en test | 0.3571 |

| Clase | Precisión | Recall | F1 |
|---|---:|---:|---:|
| Business | 0.8304 | 0.8420 | 0.8361 |
| Sci_Tech | 0.8495 | 0.8580 | 0.8537 |
| Sports | 0.9481 | 0.9500 | 0.9491 |
| World | 0.9014 | 0.8780 | 0.8896 |

## Interpretación

La validación alcanzó su menor pérdida en la época 8. A partir de allí, la pérdida de entrenamiento siguió disminuyendo, pero la de validación aumentó levemente. Se restauraron los parámetros correspondientes a la mejor época. `Sports` presenta el mejor F1 y `Business`, el menor. La evaluación test se utilizó para reportar resultados finales, no para seleccionar la época.
