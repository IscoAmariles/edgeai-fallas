# Bitácora — Sesión 01 · 14 de octubre de 2026

**Autor de la sesión:** Francisco Javier Amariles Jurado
**Duración aprox.:** 3 h (19:00–22:00)
**Objetivo:** montar el pipeline TinyOps de punta a punta y validar la comparación LSTM vs. Transformer con datos simulados, sin depender todavía de los datos reales.

---

## 1. Contexto y decisiones previas

Indicaciones del tutor (transmitidas por Jean, reunión del 14-oct-2026):

| Indicación | Implicación |
|---|---|
| "Vamos a usar Raspberry" | Plataforma definida: **Raspberry Pi** (préstamo del tutor). Inferencia en CPU ARM, sin GPU. |
| "La metodología TinyOps está bien" | Marco de desarrollo aprobado: **TinyOps (TinyMLOps)**. |
| "Toca empezar a entrenar con la data" | Se inicia la parte práctica de OE2. |
| "Validaciones con data simulada, entrenamiento con data real" | Datos **simulados para validar** el pipeline; los **datos reales** se usan para el entrenamiento definitivo. |
| "Transformer o LSTM puede estar bien, pero hay que justificar" | Se diseña un **experimento comparativo** con un criterio de selección fijado antes de ver resultados. |
| "Ya está aprobado el anteproyecto" | Sin trámite formal pendiente; falta el acceso a matrícula. |

Decisiones técnicas tomadas en la sesión:

- **Framework:** PyTorch. Exportación a **ONNX Runtime**, que corre bien en Raspberry Pi y exporta Transformers sin problemas.
- **Variables:** vibración triaxial, temperatura y diferencial de presión.
- **Clases:** `normal`, `desbalanceo`, `desalineacion`, `desgaste`.
- **Modelos:** LSTM vs. Transformer, más una **CNN-1D como línea base** (la arquitectura que menciona el anteproyecto).
- **Repositorio:** `C:\GitProjects\edgeai-fallas` (fuera de OneDrive, para evitar conflictos de sincronización con `.git`).

## 2. Qué se construyó

| Etapa TinyOps | Archivo | Qué hace |
|---|---|---|
| Datos | `src/simulate.py` | Genera grabaciones simuladas con firmas físicas conocidas: 1x radial (desbalanceo), 2x y componente axial (desalineación), impulsos a BPFO sobre una resonancia (desgaste). Temperatura y dP según `FIRMAS_AUX`. |
| Datos | `src/data.py` | Ventaneo (1024 muestras = 0,2 s a 5120 Hz, traslape del 50 %), **split por grabación** (70/15/15), normalización ajustada solo con train. |
| Modelos | `src/models.py` | Embebido por parches (32 muestras → 32 tokens), codificador intercambiable (LSTM / Transformer / CNN) y fusión con estadísticos de temperatura y dP. |
| Entrenamiento | `src/train.py` | Varias semillas, parada temprana por F1 de validación, GPU si está disponible, registro en MLflow. |
| Optimización | `src/export.py` | ONNX FP32 → pre-procesamiento → ONNX INT8 (cuantización dinámica); mide la caída de accuracy y la latencia. |
| Comparación | `src/summarize.py` | Tabla media ± desviación y aplicación del criterio de `configs/criterio_seleccion.yaml`. |
| Borde | `edge/benchmark_edge.py` | Benchmark para la Raspberry; solo requiere `numpy` y `onnxruntime`. |

**Diseño de la comparación justa.** LSTM y Transformer comparten entrada, embebido, cabeza de fusión y presupuesto de parámetros (78 k vs. 81 k). Solo cambia el codificador de secuencia, así que las diferencias de desempeño se pueden atribuir a la arquitectura.

**Criterio de selección (fijado ANTES de ver resultados):** accuracy INT8 ≥ 95 %, latencia ≤ 50 ms y, en caso de empate, gana el modelo más liviano (KB en INT8).

## 3. Comandos ejecutados

```powershell
cd C:\GitProjects\edgeai-fallas
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Verificación de GPU
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Entrenamiento: 3 modelos × 5 semillas (GPU NVIDIA GeForce RTX 4050 Laptop)
python src/train.py --config configs/lstm.yaml --seeds 0 1 2 3 4
python src/train.py --config configs/transformer.yaml --seeds 0 1 2 3 4
python src/train.py --config configs/cnn.yaml --seeds 0 1 2 3 4

# Exportación ONNX FP32 / INT8 de las 15 corridas
Get-ChildItem runs\*\seed* | ForEach-Object { python src/export.py --run $_.FullName }

# Tabla comparativa
python src/summarize.py
```

## 4. Datos usados

| | Train | Val | Test |
|---|---|---|---|
| Grabaciones simuladas | 84 | 16 | 20 |
| Ventanas | 4116 | 784 | 980 |

30 grabaciones por clase, 5 s cada una, rpm aleatorias entre 1500 y 3600, severidad aleatoria entre 0,2 y 1,0. El split por grabación usa una semilla fija (0), así que los tres modelos se evalúan sobre las mismas 20 grabaciones de test.

## 5. Resultados

### 5.1 Tabla comparativa (5 semillas por modelo, test)

| Modelo | Parámetros | Accuracy FP32 | F1-macro FP32 | Accuracy INT8 | Latencia INT8 (ms)* | Tamaño INT8 (KB) | Cumple metas |
|---|---|---|---|---|---|---|---|
| CNN-1D | 36 724 | 1,0000 ± 0,0000 | 1,0000 ± 0,0000 | 1,0000 ± 0,0000 | 0,15 ± 0,02 | 47 | Sí |
| LSTM | 78 324 | 0,9976 ± 0,0018 | 0,9976 ± 0,0018 | 0,9978 ± 0,0018 | 0,12 ± 0,01 | 90 | Sí |
| Transformer | 80 884 | 0,9980 ± 0,0017 | 0,9980 ± 0,0017 | 0,9980 ± 0,0017 | 0,19 ± 0,01 | 129 | Sí |

\* Latencia medida en la CPU del portátil (batch 1, 4 hilos). **No es la cifra de la tesis**: la válida se mide en la Raspberry.

Seleccionado por el criterio: **CNN-1D**, por desempate de tamaño, ya que los tres cumplen las metas.

### 5.2 Accuracy de test por semilla (FP32)

| Semilla | LSTM | Transformer | CNN-1D |
|---|---|---|---|
| 0 | 1,0000 | 0,9990 | 1,0000 |
| 1 | 0,9969 | 0,9980 | 1,0000 |
| 2 | 0,9969 | 0,9980 | 1,0000 |
| 3 | 0,9949 | 0,9949 | 1,0000 |
| 4 | 0,9990 | 1,0000 | 1,0000 |

### 5.3 Convergencia (épocas hasta la parada temprana, paciencia = 6)

| Modelo | s0 | s1 | s2 | s3 | s4 | Media |
|---|---|---|---|---|---|---|
| LSTM | 11 | 13 | 21 | 13 | 12 | 14,0 |
| Transformer | 10 | 7 | 7 | 7 | 11 | 8,4 |
| CNN-1D | 7 | 7 | 7 | 7 | 8 | 7,2 |

### 5.4 Efecto de la cuantización INT8 (CPU del portátil)

| Modelo | Latencia FP32 (ms) | Latencia INT8 (ms) | Tamaño FP32 → INT8 | Caída de accuracy |
|---|---|---|---|---|
| CNN-1D | 0,05–0,07 | 0,14–0,19 | 145 → 47 KB (−68 %) | 0,00 |
| LSTM | 0,11–0,15 | 0,11–0,15 | 314 → 90 KB (−71 %) | 0,00 (una semilla subió 0,10 pts) |
| Transformer | 0,15–0,16 | 0,17–0,21 | 345 → 129 KB (−63 %) | 0,00 |

## 6. Análisis

1. **El pipeline funciona de punta a punta** en el equipo de trabajo (Windows + GPU RTX 4050): simulación → entrenamiento → ONNX → INT8 → tabla. Es el objetivo de esta sesión.
2. **Los datos simulados son demasiado fáciles para discriminar entre modelos.** Los tres superan el 99,4 % y las diferencias (≤ 0,3 puntos) caen dentro de la desviación entre semillas. Este resultado **no sirve para elegir arquitectura**; solo valida el pipeline.
3. **El LSTM es menos estable al entrenar.** En las semillas 1 y 2 el F1 de validación cayó hasta 0,73–0,75 en las primeras épocas antes de recuperarse, y en promedio necesitó 14 épocas, contra 8,4 del Transformer. A confirmar con datos reales.
4. **La cuantización INT8 no costó precisión** en ningún modelo y redujo el tamaño entre 63 y 71 %.
5. **INT8 no aceleró en la CPU x86 del portátil** (en la CNN fue incluso más lento). Con modelos tan pequeños, el costo de convertir a INT8 y de vuelta pesa más que el ahorro. En la CPU ARM de la Raspberry el comportamiento puede ser distinto: **se debe medir allá**, que es el principio central de TinyOps.
6. **Las metas de 95 % y 50 ms tienen mucho margen** en el portátil (latencias por debajo de 0,2 ms). Aunque la Raspberry fuera 50 veces más lenta, seguirían dentro de la meta. Hay que confirmarlo en el hardware.

## 7. Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| `MLflow falló … DLL load failed while importing _collections_cy` + alerta "Part of this app has been blocked" | El **Control inteligente de aplicaciones** de Windows bloquea una DLL sin firma de SQLAlchemy (dependencia de MLflow). | `train.py` fija `DISABLE_SQLALCHEMY_CEXT_RUNTIME=1` (SQLAlchemy en Python puro). Si aun así falla, usa el almacén en carpeta `mlruns/`. El entrenamiento nunca se interrumpe por MLflow. |
| MLflow rechaza el almacén `./mlruns` | Las versiones nuevas de MLflow exigen una base de datos. | Se usa `sqlite:///mlflow.db`. |
| Entrenaba en CPU teniendo GPU | El código original no movía el modelo a CUDA. | `train.py --device auto`: usa la GPU para entrenar y siempre mide la latencia en CPU. |
| Aviso "consider to run pre-processing before quantization" | Faltaba simplificar el grafo antes de cuantizar. | `export.py` aplica `quant_pre_process`. El tamaño INT8 del Transformer bajó de 143 a 129 KB. |
| `DeprecationWarning` del exportador TorchScript | PyTorch ≥ 2.9 cambió el exportador por defecto. | Se mantiene el exportador clásico (`dynamo=False`) porque funciona y es más estable con Transformers. Migrar cuando se elimine. |
| `UserWarning` de LSTM y batch size en ONNX | Advertencia genérica del exportador para RNN. | Sin efecto: el modelo se exporta y se usa con batch 1. |
| Las librerías se instalaron en el Python de la Microsoft Store | El entorno `.venv` no estaba activado al instalar. | Activar `.venv\Scripts\activate` antes de `pip install`. |

## 8. Pendientes

- [ ] Subir el repositorio a GitHub e invitar a Jean como colaborador.
- [ ] Acordar con Jean el **formato de los datos reales** (propuesta en `cargar_csv`, `src/data.py`).
- [ ] Validar con el equipo los **supuestos de temperatura y dP** por clase (`FIRMAS_AUX` en `src/simulate.py`).
- [ ] Preguntar al tutor por la Raspberry: modelo, RAM, si tiene AI HAT, fecha de entrega y tiempo de préstamo.
- [ ] Ajustar en la tesis el OE1: la comparación Jetson vs. Raspberry pasa a ser documental, o a Raspberry vs. PC/nube.

## 9. Próxima sesión

**Escenario simulado difícil**: severidad baja (0,05–0,3), más ruido, rpm fuera del rango de entrenamiento y fallas combinadas. Objetivo: que los modelos se diferencien y evaluar su robustez antes de tener los datos reales.
