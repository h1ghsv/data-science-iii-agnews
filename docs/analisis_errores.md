# Análisis de errores del clasificador AG News

## Resultados del test final

Sobre 2,000 noticias del test reservado, se clasificaron correctamente 1,764 y se registraron 236 errores (Accuracy: 88.20 %).

### Rendimiento por categoría

| Categoría | Noticias | Aciertos | Accuracy |
|---|---:|---:|---:|
| Business | 500 | 421 | 84.20 % |
| Sci_Tech | 500 | 429 | 85.80 % |
| Sports | 500 | 475 | 95.00 % |
| World | 500 | 439 | 87.80 % |

### Confusiones principales

| Real | Predicha | Casos |
|---|---|---:|
| Business | Sci_Tech | 49 |
| Sci_Tech | Business | 49 |
| World | Business | 28 |
| Business | World | 26 |
| World | Sci_Tech | 19 |
| Sci_Tech | World | 14 |
| World | Sports | 14 |
| Sports | Business | 9 |

### Influencia de la longitud

| Tokens | Documentos | Accuracy |
|---|---:|---:|
| 13–33 | 563 | 85.97 % |
| 34–38 | 451 | 90.69 % |
| 39–44 | 555 | 88.11 % |
| 45–137 | 431 | 88.63 % |

![Aciertos por categoría y longitud](../outputs/analisis_errores.png)

### Interpretación

La categoría con mayor proporción de aciertos entre sus noticias fue **Sports** (95.0 % de sus ejemplos correctos). La más difícil fue **Business** (84.2 %). La confusión más frecuente fue **Business → Sci_Tech** (49 noticias). También se observaron **Sci_Tech → Business** (49 casos).

La variación de aciertos por longitud es descriptiva y no demuestra causalidad. `errores_alta_confianza.csv` identifica las predicciones incorrectas más seguras para una revisión cualitativa posterior, sin repetir el texto original de las noticias.

**Nota metodológica:** estos resultados provienen del test utilizado una sola vez tras seleccionar la mejor época con validación. El análisis de errores es posterior y no se utiliza para cambiar hiperparámetros ni volver a entrenar con información de test. Las divisiones por longitud se calcularon únicamente para describir los resultados.
