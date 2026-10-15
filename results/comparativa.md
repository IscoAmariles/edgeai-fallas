| Experimento | Modelo | Semillas | Parámetros | Accuracy FP32 | F1-macro FP32 | Accuracy INT8 | Latencia INT8 (ms) | Tamaño INT8 (KB) | Cumple metas |
|---|---|---|---|---|---|---|---|---|---|
| cnn_sim | cnn | 3 | 36,724 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.25 ± 0.02 | 47 ± 0 | Sí |
| lstm_sim | lstm | 3 | 78,324 | 0.9976 ± 0.0019 | 0.9976 ± 0.0019 | 0.9976 ± 0.0019 | 0.18 ± 0.01 | 96 ± 0 | Sí |
| transformer_sim | transformer | 3 | 80,884 | 0.9993 ± 0.0010 | 0.9993 ± 0.0010 | 0.9993 ± 0.0010 | 0.25 ± 0.05 | 143 ± 0 | Sí |

Criterio: accuracy INT8 ≥ 0.95 y latencia INT8 ≤ 50 ms; desempate por `kb_int8` (menor gana).
**Seleccionado:** cnn_sim

_Nota: latencias medidas en CPU local salvo que se indique; la cifra final se mide en la Raspberry._