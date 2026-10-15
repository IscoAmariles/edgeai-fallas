"""
Resume todas las corridas en una tabla comparativa (media ± desviación por modelo) y aplica
el CRITERIO DE SELECCIÓN definido ANTES de ver resultados (configs/criterio_seleccion.yaml).

Uso:
    python src/summarize.py
Salida: results/comparativa.csv y results/comparativa.md (lista para pegar en la tesis).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]


def main():
    crit = yaml.safe_load((RAIZ / "configs" / "criterio_seleccion.yaml").read_text(encoding="utf-8"))
    filas = defaultdict(list)
    for mf in sorted((RAIZ / "runs").glob("*/seed*/metrics.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        ex = mf.parent / "export.json"
        e = json.loads(ex.read_text()) if ex.exists() else None
        filas[m["experimento"]].append((m, e))
    if not filas:
        print("No hay corridas en runs/. Ejecuta primero src/train.py")
        return

    def ms(xs):
        xs = [x for x in xs if x is not None]
        return (np.mean(xs), np.std(xs)) if xs else (np.nan, np.nan)

    out = []
    for ex, rs in filas.items():
        acc = ms([r[0]["test"]["accuracy"] for r in rs])
        f1 = ms([r[0]["test"]["f1_macro"] for r in rs])
        acc8 = ms([r[1]["int8"]["accuracy"] if r[1] else None for r in rs])
        lat8 = ms([r[1]["int8"]["latencia_mediana_ms"] if r[1] else None for r in rs])
        kb8 = ms([r[1]["int8"]["tamano_kb"] if r[1] else None for r in rs])
        cumple = (acc8[0] >= crit["accuracy_minima"]) and (lat8[0] <= crit["latencia_maxima_ms"])
        out.append({"experimento": ex, "modelo": rs[0][0]["modelo"], "n_semillas": len(rs),
                    "parametros": rs[0][0]["parametros"], "acc": acc, "f1": f1, "acc_int8": acc8,
                    "lat_int8_ms": lat8, "kb_int8": kb8, "cumple": bool(cumple)})

    f = lambda t, d=4: f"{t[0]:.{d}f} ± {t[1]:.{d}f}" if not np.isnan(t[0]) else "—"
    md = ["| Experimento | Modelo | Semillas | Parámetros | Accuracy FP32 | F1-macro FP32 | Accuracy INT8 | Latencia INT8 (ms) | Tamaño INT8 (KB) | Cumple metas |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['experimento']} | {r['modelo']} | {r['n_semillas']} | {r['parametros']:,} | {f(r['acc'])} | {f(r['f1'])} | "
                  f"{f(r['acc_int8'])} | {f(r['lat_int8_ms'], 2)} | {f(r['kb_int8'], 0)} | {'Sí' if r['cumple'] else 'No'} |")
    candidatos = sorted([r for r in out if r["cumple"]], key=lambda r: r[crit["desempate"]][0])
    md.append("")
    md.append(f"Criterio: accuracy INT8 ≥ {crit['accuracy_minima']} y latencia INT8 ≤ {crit['latencia_maxima_ms']} ms; desempate por `{crit['desempate']}` (menor gana).")
    md.append(f"**Seleccionado:** {candidatos[0]['experimento'] if candidatos else 'ninguno cumple aún'}")
    md.append("\n_Nota: latencias medidas en CPU local salvo que se indique; la cifra final se mide en la Raspberry._")

    (RAIZ / "results").mkdir(exist_ok=True)
    (RAIZ / "results" / "comparativa.md").write_text("\n".join(md), encoding="utf-8")
    import csv
    with open(RAIZ / "results" / "comparativa.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["experimento", "modelo", "semillas", "parametros", "acc_media", "acc_std", "f1_media", "f1_std",
                    "acc_int8_media", "lat_int8_ms_media", "kb_int8_media", "cumple"])
        for r in out:
            w.writerow([r["experimento"], r["modelo"], r["n_semillas"], r["parametros"], *r["acc"], *r["f1"],
                        r["acc_int8"][0], r["lat_int8_ms"][0], r["kb_int8"][0], r["cumple"]])
    print("\n".join(md))


if __name__ == "__main__":
    main()
