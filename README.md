# edgeai-fallas

Clasificación de fallas en maquinaria industrial con modelos de IA desplegados en **Raspberry Pi**.
Trabajo de grado — Maestría en IA y Ciencia de Datos, UAO.
Autores: Francisco Javier Amariles Jurado y Jean Pierre Henriquez Ebratt.

Variables: **vibración triaxial**, **temperatura** y **diferencial de presión**.
Clases: `normal`, `desbalanceo`, `desalineacion`, `desgaste`.

## Metodología: TinyOps (TinyMLOps)

El resultado que vale es **el medido en el dispositivo de borde**, no en el PC. Cada etapa deja un artefacto versionado:

| Etapa | Script | Artefacto |
|---|---|---|
| 1. Datos | `src/simulate.py`, `src/data.py` | grabaciones, splits por grabación, normalización |
| 2. Entrenamiento | `src/train.py` | `model.pt`, `metrics.json` (+ MLflow) |
| 3. Optimización | `src/export.py` | `model_fp32.onnx`, `model_int8.onnx`, caída por cuantización |
| 4. Despliegue y 5. Medición en borde | `edge/benchmark_edge.py` | `edge_benchmark.json` (latencia real en la Raspberry) |
| Comparación | `src/summarize.py` | `results/comparativa.md` (tabla para la tesis) |

## Decisiones de diseño (justificación para la tesis)

1. **Comparación justa LSTM vs. Transformer.** Ambos comparten la misma entrada, el mismo embebido por parches y la misma fusión con temperatura y dP. Solo cambia el codificador de secuencia, con un número de parámetros similar (~80 k). La CNN-1D se incluye como línea base, que es la que prometió el anteproyecto.
2. **Split por grabación, no por ventana**, con semilla fija. Así se evita la fuga de datos y todos los modelos se evalúan sobre las mismas grabaciones.
3. **Normalización calculada solo con train.**
4. **Varias semillas por modelo** (`--seeds 0 1 2 3 4`). Se reporta media ± desviación.
5. **Criterio de selección fijado antes de ver resultados** en `configs/criterio_seleccion.yaml`: accuracy INT8 ≥ 95 %, latencia ≤ 50 ms y, si hay empate, gana el más liviano.
6. **Parches de 32 muestras.** Una ventana de 1024 muestras se convierte en 32 tokens, lo que mantiene acotado el costo de la atención en la CPU de la Raspberry.
7. **Datos simulados = validación** (indicación del tutor). Sirven para verificar el pipeline y que el modelo responde a firmas físicas conocidas. El entrenamiento definitivo se hace con datos reales.

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Raspberry: source .venv/bin/activate)
pip install -r requirements.txt

python src/train.py --config configs/lstm.yaml --seeds 0 1 2 3 4
python src/train.py --config configs/transformer.yaml --seeds 0 1 2 3 4
python src/train.py --config configs/cnn.yaml --seeds 0 1 2 3 4

# exportar cada corrida a ONNX FP32/INT8 (PowerShell)
Get-ChildItem runs\*\seed* | ForEach-Object { python src/export.py --run $_.FullName }

python src/summarize.py                                   # -> results/comparativa.md
mlflow ui --backend-store-uri sqlite:///mlflow.db         # -> http://127.0.0.1:5000
```

En la Raspberry (solo necesita `pip install numpy onnxruntime`): copia `edge/benchmark_edge.py` y una carpeta `runs/<exp>/seed<k>/`, y ejecuta:

```bash
python benchmark_edge.py --run seed0 --hilos 1 2 4
```

## Pendientes

- [ ] Cargador de datos reales (`cargar_csv` en `src/data.py`). Hay que acordar el formato con Jean.
- [ ] Validar con el equipo los supuestos de temperatura y dP por clase (`FIRMAS_AUX` en `src/simulate.py`).
- [ ] Medir en la Raspberry: modelo, RAM y si tiene AI HAT.
- [ ] Escenario de validación más difícil (baja severidad, más ruido, rpm fuera del rango de entrenamiento).

## Estructura

```
configs/   experimentos (.yaml) y criterio de selección
src/       simulate, data, models, train, export, summarize
edge/      benchmark para la Raspberry Pi
data/      datos reales (NO se suben a git)
results/   tablas y figuras para la tesis
bitacora/  registro por sesión (comandos, resultados, decisiones)
runs/      corridas (ignorado en git)
```
