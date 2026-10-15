"""
Benchmark EN LA RASPBERRY PI. Solo necesita numpy y onnxruntime:
    pip install numpy onnxruntime

Copia a la Raspberry la carpeta runs/<experimento>/seed<k>/ (los .onnx y test_set.npz) y ejecuta:
    python benchmark_edge.py --run ruta/a/seed0
    python benchmark_edge.py --run ruta/a/seed0 --hilos 1 2 4

Estas son las cifras de latencia y accuracy que van al capítulo de resultados.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def medir(ruta, d, hilos, n_lat):
    so = ort.SessionOptions()
    so.intra_op_num_threads = hilos
    s = ort.InferenceSession(str(ruta), so, providers=["CPUExecutionProvider"])
    vib, aux, y = d["vib"], d["aux"], d["y"]
    pred = np.array([s.run(None, {"vib": vib[i:i + 1], "aux": aux[i:i + 1]})[0].argmax() for i in range(len(y))])
    for i in range(30):
        s.run(None, {"vib": vib[:1], "aux": aux[:1]})
    t = []
    for i in range(n_lat):
        j = i % len(y)
        t0 = time.perf_counter()
        s.run(None, {"vib": vib[j:j + 1], "aux": aux[j:j + 1]})
        t.append((time.perf_counter() - t0) * 1000)
    return {"accuracy": float((pred == y).mean()), "lat_mediana_ms": float(np.median(t)),
            "lat_p95_ms": float(np.percentile(t, 95)), "lat_max_ms": float(np.max(t))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hilos", type=int, nargs="+", default=[4])
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    run = Path(a.run)
    d = np.load(run / "test_set.npz")
    res = {"equipo": platform.platform(), "maquina": platform.machine(), "onnxruntime": ort.__version__, "resultados": []}
    for prec in ("fp32", "int8"):
        for h in a.hilos:
            r = medir(run / f"model_{prec}.onnx", d, h, a.n)
            r.update({"precision": prec, "hilos": h})
            res["resultados"].append(r)
            print(f"{prec} | {h} hilos | acc {r['accuracy']:.4f} | mediana {r['lat_mediana_ms']:.2f} ms | p95 {r['lat_p95_ms']:.2f} ms")
    (run / "edge_benchmark.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
