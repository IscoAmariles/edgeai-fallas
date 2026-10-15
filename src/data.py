"""
Carga, ventaneo, split por grabación y normalización.

Regla TinyOps clave: el split se hace POR GRABACIÓN, nunca por ventana. Si ventanas de una misma
grabación caen en train y en test, la precisión sale inflada (fuga de datos / data leakage).

Para usar datos reales basta con escribir un cargador que devuelva una lista de `Grabacion`
(ver simulate.py) y registrarlo en `cargar_grabaciones`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from simulate import CLASES, Grabacion, generar_grabaciones


def cargar_grabaciones(cfg: dict) -> list[Grabacion]:
    fuente = cfg["fuente"]
    if fuente == "simulado":
        return generar_grabaciones(**cfg.get("simulado", {}))
    if fuente == "csv":
        return cargar_csv(cfg["csv"])
    raise ValueError(f"Fuente de datos desconocida: {fuente}")


def cargar_csv(cfg: dict) -> list[Grabacion]:
    """Pendiente: cargador de datos reales.

    Formato propuesto (acordar con Jean): una carpeta por grabación con
      vib.csv  -> columnas ax, ay, az   (fs_vib Hz)
      aux.csv  -> columnas temp_c, dp_bar (fs_aux Hz)
      meta.json -> {"etiqueta": "desgaste", "rpm": 1800, ...}
    """
    raise NotImplementedError("Implementar cuando esté definido el formato de los datos reales.")


def split_por_grabacion(grabs: list[Grabacion], fracciones=(0.7, 0.15, 0.15), semilla: int = 0):
    """Split estratificado por clase, a nivel de grabación."""
    rng = np.random.default_rng(semilla)
    tr, va, te = [], [], []
    for c in sorted({g.etiqueta for g in grabs}):
        gc = [g for g in grabs if g.etiqueta == c]
        idx = rng.permutation(len(gc))
        n_tr = int(round(fracciones[0] * len(gc)))
        n_va = int(round(fracciones[1] * len(gc)))
        tr += [gc[i] for i in idx[:n_tr]]
        va += [gc[i] for i in idx[n_tr : n_tr + n_va]]
        te += [gc[i] for i in idx[n_tr + n_va :]]
    ids = [set(g.id for g in s) for s in (tr, va, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "Fuga: grabación repetida"
    return tr, va, te


def ventanear(grabs: list[Grabacion], largo: int, salto: int):
    """Corta ventanas sincronizadas de vibración y auxiliares.

    Devuelve vib (N, 3, largo), aux (N, 2, largo_aux), y (N,), ids (N,)
    """
    V, A, Y, I = [], [], [], []
    for g in grabs:
        ratio = g.fs_aux / g.fs_vib
        largo_aux = max(int(round(largo * ratio)), 1)
        for ini in range(0, g.vib.shape[1] - largo + 1, salto):
            ia = int(ini * ratio)
            a = g.aux[:, ia : ia + largo_aux]
            if a.shape[1] < largo_aux:  # borde final
                a = np.pad(a, ((0, 0), (0, largo_aux - a.shape[1])), mode="edge")
            V.append(g.vib[:, ini : ini + largo])
            A.append(a)
            Y.append(g.etiqueta)
            I.append(g.id)
    return np.stack(V), np.stack(A), np.array(Y, dtype=np.int64), np.array(I)


class Normalizador:
    """Media/desviación por canal, calculadas SOLO con train (otra fuente común de fuga)."""

    def ajustar(self, vib, aux):
        self.mv = vib.mean(axis=(0, 2), keepdims=True)[0]
        self.sv = vib.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        self.ma = aux.mean(axis=(0, 2), keepdims=True)[0]
        self.sa = aux.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        return self

    def aplicar(self, vib, aux):
        return ((vib - self.mv) / self.sv).astype(np.float32), ((aux - self.ma) / self.sa).astype(np.float32)

    def a_dict(self):
        return {k: getattr(self, k).tolist() for k in ("mv", "sv", "ma", "sa")}


class VentanasDS(Dataset):
    def __init__(self, vib, aux, y):
        self.vib, self.aux, self.y = torch.from_numpy(vib), torch.from_numpy(aux), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.vib[i], self.aux[i], self.y[i]


def preparar(cfg: dict, semilla_split: int = 0):
    """Pipeline completo de datos. Devuelve datasets y metadatos."""
    grabs = cargar_grabaciones(cfg["datos"])
    tr, va, te = split_por_grabacion(grabs, semilla=semilla_split)
    w = cfg["ventana"]
    conj = {n: ventanear(s, w["largo"], w["salto"]) for n, s in (("train", tr), ("val", va), ("test", te))}
    norm = Normalizador().ajustar(conj["train"][0], conj["train"][1])
    ds = {}
    for n, (v, a, y, _) in conj.items():
        v, a = norm.aplicar(v, a)
        ds[n] = VentanasDS(v, a, y)
    info = {
        "clases": CLASES,
        "n_grabaciones": {"train": len(tr), "val": len(va), "test": len(te)},
        "n_ventanas": {n: len(d) for n, d in ds.items()},
        "forma_vib": list(ds["train"].vib.shape[1:]),
        "forma_aux": list(ds["train"].aux.shape[1:]),
        "normalizacion": norm.a_dict(),
    }
    return ds, info
