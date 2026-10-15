"""
Exportación TinyOps: PyTorch -> ONNX FP32 -> ONNX INT8 (cuantización dinámica),
y medición de cuánta precisión se pierde al cuantizar.

Uso:
    python src/export.py --run runs/transformer_sim/seed0

Genera en esa carpeta:
    model_fp32.onnx, model_int8.onnx, test_set.npz, export.json
Los .onnx y test_set.npz son lo que se copia a la Raspberry para edge/benchmark_edge.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import accuracy_score, f1_score

from data import preparar
from models import ClasificadorFallas


def evaluar_onnx(ruta, vib, aux, y, hilos=4, n_lat=200):
    so = ort.SessionOptions()
    so.intra_op_num_threads = hilos
    s = ort.InferenceSession(str(ruta), so, providers=["CPUExecutionProvider"])
    preds = [int(s.run(None, {"vib": vib[i : i + 1], "aux": aux[i : i + 1]})[0].argmax()) for i in range(len(y))]
    for i in range(20):
        s.run(None, {"vib": vib[:1], "aux": aux[:1]})
    t = []
    for i in range(n_lat):
        j = i % len(y)
        t0 = time.perf_counter()
        s.run(None, {"vib": vib[j : j + 1], "aux": aux[j : j + 1]})
        t.append((time.perf_counter() - t0) * 1000)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "latencia_mediana_ms": float(np.median(t)),
        "latencia_p95_ms": float(np.percentile(t, 95)),
        "tamano_kb": Path(ruta).stat().st_size / 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="carpeta runs/<experimento>/seed<k>")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    info = json.loads((run / "data_info.json").read_text(encoding="utf-8"))

    m = cfg["modelo"]
    modelo = ClasificadorFallas(m["tipo"], info["forma_vib"], info["forma_aux"], len(info["clases"]), **m.get("params", {}))
    modelo.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    modelo.eval()

    ds, _ = preparar(cfg, semilla_split=cfg.get("semilla_split", 0))
    te = ds["test"]
    vib, aux, y = te.vib.numpy(), te.aux.numpy(), te.y.numpy()
    np.savez_compressed(run / "test_set.npz", vib=vib, aux=aux, y=y, clases=np.array(info["clases"]))

    fp32 = run / "model_fp32.onnx"
    torch.onnx.export(
        modelo, (te.vib[:1], te.aux[:1]), str(fp32),
        input_names=["vib", "aux"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    # Pre-procesamiento recomendado por onnxruntime antes de cuantizar (fusiona y simplifica el grafo)
    prep = run / "model_fp32_prep.onnx"
    try:
        quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
        origen = prep
    except Exception as e:
        print(f"(pre-procesamiento omitido: {e})")
        origen = fp32
    int8 = run / "model_int8.onnx"
    quantize_dynamic(str(origen), str(int8), weight_type=QuantType.QInt8)

    res = {"fp32": evaluar_onnx(fp32, vib, aux, y), "int8": evaluar_onnx(int8, vib, aux, y)}
    res["caida_accuracy_int8"] = res["fp32"]["accuracy"] - res["int8"]["accuracy"]
    (run / "export.json").write_text(json.dumps(res, indent=2))
    for k in ("fp32", "int8"):
        r = res[k]
        print(f"{k}: acc {r['accuracy']:.4f} | f1 {r['f1_macro']:.4f} | {r['latencia_mediana_ms']:.2f} ms | {r['tamano_kb']:.0f} KB")
    print(f"Caída de accuracy por INT8: {res['caida_accuracy_int8']*100:.2f} puntos")


if __name__ == "__main__":
    main()
