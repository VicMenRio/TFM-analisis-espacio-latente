# TFM — Estudio Comparativo del Espacio Latente de Modelos Generativos de Texto-Imagen

Repositorio del Trabajo de Fin de Máster. Analiza y contrasta la estructura interna del espacio latente de tres modelos de difusión de código abierto (SD 1.5, SD 2.1, SDXL) en tres ejes: geometría, semántica y respuesta al ajuste fino.

## Estructura

```
repositorio/
├── setup/                   Entorno Colab y dependencias
├── modelos/                 Carga de modelos y validación inicial (roundtrip VAE)
├── corpus/                  Generación del corpus de latents (150 por modelo)
├── analisis_geometrico/     Separabilidad, dimensionalidad efectiva, correlación latente↔imagen
├── analisis_semantico/      Direcciones semánticas (color, iluminación, estilo), PCA, entrelazado
├── ajuste_fino/             Fine-tuning LoRA sobre WikiArt + análisis de deriva pre/post
```

Cada sección contiene:
- Scripts / Notebooks de Colab que ejecutan el análisis
- La carpeta `resultados/` con figuras (`.png`) y métricas (`.json`)


## Orden de ejecución

1. `setup/entorno.ipynb` — monta Drive e instala dependencias
2. `modelos/validacion.ipynb` — carga los tres modelos y verifica el ciclo encode/decode
3. `corpus/generacion_corpus.ipynb` — genera los 150 latents por modelo
4. `analisis_geometrico/geometrica.py`
5. `analisis_semantico/semantica.py`
6. `ajuste_fino/` — en orden: `preparacion_dataset` → `entrenamiento` → `captura_latents` → `analisis_deriva`

## Modelos

| Modelo | ID HuggingFace |
|---|---|
| SD 1.5 | `runwayml/stable-diffusion-v1-5` |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |

## Entorno

Experimentos ejecutados en Google Colab (GPU T4, 15 GB VRAM).
Dependencias: ver `setup/requirements_colab.txt`.
