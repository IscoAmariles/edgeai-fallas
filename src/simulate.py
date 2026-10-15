"""
Generador de datos simulados multimodales (vibración triaxial + temperatura + diferencial de presión).

Propósito (según indicación del tutor): VALIDAR el pipeline y el comportamiento de los modelos
con señales cuyo origen físico conocemos. NO reemplaza los datos reales de entrenamiento.

Cada "grabación" simula una máquina rotativa operando en condiciones fijas (rpm, severidad, ruido)
durante `duracion_s` segundos. Las ventanas se cortan después, en data.py, y el split se hace
por grabación para evitar fuga de datos.

Firmas de falla usadas (literatura clásica de análisis de vibraciones):
- normal       : 1x de baja amplitud + ruido.
- desbalanceo  : 1x dominante en los ejes radiales (x, y), desfasados ~90°.
- desalineacion: 2x dominante y componente axial (z) alta; 1x moderado.
- desgaste     : impulsos periódicos a la frecuencia de defecto de pista externa (BPFO)
                 que excitan una resonancia estructural (~2.5-3.5 kHz).

SUPUESTOS A VALIDAR CON EL EQUIPO (temperatura y diferencial de presión):
- desgaste      -> aumento de temperatura (fricción) y leve caída de dP.
- desalineacion -> aumento moderado de temperatura.
- desbalanceo   -> aumento leve de temperatura, dP con más fluctuación.
Los valores están en FIRMAS_AUX y se pueden ajustar sin tocar el resto del código.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASES = ["normal", "desbalanceo", "desalineacion", "desgaste"]

# (delta_temperatura_C, delta_dP_bar, std_extra_dP_bar) por clase, a severidad 1.0
FIRMAS_AUX = {
    "normal": (0.0, 0.00, 0.000),
    "desbalanceo": (2.0, 0.00, 0.020),
    "desalineacion": (4.0, -0.02, 0.005),
    "desgaste": (8.0, -0.06, 0.005),
}


@dataclass
class Grabacion:
    """Una grabación continua. Es la unidad para hacer splits (nunca se parte entre conjuntos)."""

    vib: np.ndarray  # (3, n_vib)  aceleración en g, ejes x, y, z
    aux: np.ndarray  # (2, n_aux)  [temperatura °C, diferencial de presión bar]
    etiqueta: int
    id: str
    fs_vib: float
    fs_aux: float
    meta: dict


def _vibracion(clase: str, rpm: float, sev: float, fs: float, n: int, rng) -> np.ndarray:
    t = np.arange(n) / fs
    f1 = rpm / 60.0
    ph = rng.uniform(0, 2 * np.pi, size=6)
    x = np.zeros((3, n))

    # Línea base común: 1x pequeño y armónicos de la máquina sana
    base = 0.05
    x[0] += base * np.sin(2 * np.pi * f1 * t + ph[0])
    x[1] += base * np.sin(2 * np.pi * f1 * t + ph[1])
    x[2] += 0.3 * base * np.sin(2 * np.pi * f1 * t + ph[2])

    if clase == "desbalanceo":
        a = 0.15 + 0.6 * sev
        x[0] += a * np.sin(2 * np.pi * f1 * t + ph[3])
        x[1] += a * np.sin(2 * np.pi * f1 * t + ph[3] + np.pi / 2)  # desfase 90°
        x[2] += 0.1 * a * np.sin(2 * np.pi * f1 * t + ph[4])
    elif clase == "desalineacion":
        a = 0.12 + 0.5 * sev
        for eje, g in ((0, 0.6), (1, 0.6), (2, 1.0)):
            x[eje] += g * a * np.sin(2 * np.pi * 2 * f1 * t + ph[eje + 3])
            x[eje] += 0.4 * g * a * np.sin(2 * np.pi * f1 * t + ph[eje])
            x[eje] += 0.2 * g * a * np.sin(2 * np.pi * 3 * f1 * t + ph[eje])
    elif clase == "desgaste":
        # Rodamiento genérico: BPFO ≈ 3.58 x f1 (aprox. rodamiento 6205)
        bpfo = 3.58 * f1
        f_res = rng.uniform(2500, 3500)
        amort = 600.0
        periodo = 1.0 / bpfo
        imp = np.zeros(n)
        k = 0
        while True:
            t0 = k * periodo + rng.normal(0, 0.01 * periodo)  # leve deslizamiento
            if t0 >= t[-1]:
                break
            i0 = max(int(t0 * fs), 0)
            tt = t[i0:] - t[i0]
            lim = min(len(tt), int(0.01 * fs))
            imp[i0 : i0 + lim] += np.exp(-amort * tt[:lim]) * np.sin(2 * np.pi * f_res * tt[:lim])
            k += 1
        a = 0.2 + 1.0 * sev
        x[0] += a * imp
        x[1] += 0.7 * a * imp
        x[2] += 0.3 * a * imp

    snr_ruido = rng.uniform(0.03, 0.12)
    x += rng.normal(0, snr_ruido, size=x.shape)
    return x.astype(np.float32)


def _auxiliares(clase: str, sev: float, fs: float, n: int, rng) -> np.ndarray:
    t = np.arange(n) / fs
    dT, ddp, sdp = FIRMAS_AUX[clase]
    T0 = rng.uniform(38, 50)  # temperatura ambiente/operación variable entre máquinas
    dp0 = rng.uniform(0.9, 1.3)  # bar
    temp = T0 + sev * dT * (1 + 0.1 * t / max(t[-1], 1e-6)) + rng.normal(0, 0.3, n)
    dp = dp0 + sev * ddp + rng.normal(0, 0.01 + sev * sdp, n)
    return np.stack([temp, dp]).astype(np.float32)


def generar_grabaciones(
    n_por_clase: int = 30,
    duracion_s: float = 5.0,
    fs_vib: float = 5120.0,
    fs_aux: float = 10.0,
    rpm_rango=(1500, 3600),
    sev_rango=(0.2, 1.0),
    semilla: int = 0,
) -> list[Grabacion]:
    rng = np.random.default_rng(semilla)
    grabs = []
    n_vib = int(duracion_s * fs_vib)
    n_aux = int(duracion_s * fs_aux)
    for etq, clase in enumerate(CLASES):
        for i in range(n_por_clase):
            rpm = rng.uniform(*rpm_rango)
            sev = 0.0 if clase == "normal" else rng.uniform(*sev_rango)
            grabs.append(
                Grabacion(
                    vib=_vibracion(clase, rpm, sev, fs_vib, n_vib, rng),
                    aux=_auxiliares(clase, sev, fs_aux, n_aux, rng),
                    etiqueta=etq,
                    id=f"sim_{clase}_{i:03d}",
                    fs_vib=fs_vib,
                    fs_aux=fs_aux,
                    meta={"rpm": float(rpm), "severidad": float(sev)},
                )
            )
    return grabs


if __name__ == "__main__":
    gs = generar_grabaciones(n_por_clase=2)
    for g in gs:
        print(g.id, g.vib.shape, g.aux.shape, CLASES[g.etiqueta], g.meta)
