"""
Entrenamiento reproducible a partir de un archivo de configuración.

Uso:
    python src/train.py --config configs/lstm.yaml
    python src/train.py --config configs/transformer.yaml --seeds 0 1 2 3 4

Por cada semilla guarda en runs/<experimento>/seed<k>/:
    model.pt, metrics.json, config.yaml, data_info.json
Si MLflow está instalado y `mlflow: true` en el config, también registra la corrida
(ver resultados con:  mlflow ui --backend-store-uri sqlite:///mlflow.db  -> http://127.0.0.1:5000).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import preparar
from models import ClasificadorFallas, contar_parametros

RAIZ = Path(__file__).resolve().parents[1]


def fijar_semilla(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def evaluar(modelo, loader, disp="cpu"):
    modelo.eval()
    ys, ps = [], []
    for v, a, y in loader:
        ps.append(modelo(v.to(disp), a.to(disp)).argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro")),
        "matriz_confusion": confusion_matrix(y, p).tolist(),
    }


@torch.no_grad()
def latencia_cpu_ms(modelo, ds, n: int = 200, hilos: int = 4):
    """Latencia por inferencia (batch 1) en la CPU local. Es una APROXIMACIÓN para ordenar
    modelos; la cifra válida para la tesis es la medida en la Raspberry (edge/benchmark_edge.py)."""
    torch.set_num_threads(hilos)
    modelo.eval()
    v, a, _ = ds[0]
    v, a = v.unsqueeze(0), a.unsqueeze(0)
    for _ in range(20):
        modelo(v, a)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        modelo(v, a)
        t.append((time.perf_counter() - t0) * 1000)
    return {"mediana_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def elegir_dispositivo(pedido: str = "auto") -> str:
    """GPU (CUDA) solo para ENTRENAR. La latencia siempre se mide en CPU, como en la Raspberry."""
    if pedido == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pedido


def entrenar_una(cfg, ds, info, semilla: int, salida: Path, disp: str = "cpu"):
    fijar_semilla(semilla)
    m = cfg["modelo"]
    modelo = ClasificadorFallas(m["tipo"], info["forma_vib"], info["forma_aux"], len(info["clases"]), **m.get("params", {})).to(disp)
    e = cfg["entrenamiento"]
    g = torch.Generator().manual_seed(semilla)
    dl_tr = DataLoader(ds["train"], batch_size=e["batch"], shuffle=True, generator=g)
    dl_va = DataLoader(ds["val"], batch_size=256)
    dl_te = DataLoader(ds["test"], batch_size=256)
    opt = torch.optim.AdamW(modelo.parameters(), lr=e["lr"], weight_decay=e.get("weight_decay", 1e-2))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=e["epocas"])
    perdida = nn.CrossEntropyLoss()

    mejor, mejor_estado, paciencia, historia = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(e["epocas"]):
        modelo.train()
        tot = 0.0
        for v, a, y in dl_tr:
            v, a, y = v.to(disp), a.to(disp), y.to(disp)
            opt.zero_grad()
            l = perdida(modelo(v, a), y)
            l.backward()
            opt.step()
            tot += l.item() * len(y)
        sched.step()
        val = evaluar(modelo, dl_va, disp)
        historia.append({"epoca": ep, "loss": tot / len(ds["train"]), "val_f1": val["f1_macro"]})
        print(f"  [{m['tipo']} s{semilla}] ep {ep:02d} loss {historia[-1]['loss']:.4f} val_f1 {val['f1_macro']:.4f}")
        if val["f1_macro"] > mejor:
            mejor, paciencia = val["f1_macro"], 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            paciencia += 1
            if paciencia >= e.get("paciencia", 5):
                print("  parada temprana")
                break
    modelo.load_state_dict(mejor_estado)
    modelo = modelo.cpu()  # evaluación final y latencia en CPU (como en el borde)

    test = evaluar(modelo, dl_te)
    metr = {
        "experimento": cfg["nombre"],
        "modelo": m["tipo"],
        "semilla": semilla,
        "parametros": contar_parametros(modelo),
        "tamano_fp32_kb": contar_parametros(modelo) * 4 / 1024,
        "mejor_val_f1": mejor,
        "test": test,
        "latencia_cpu_local": latencia_cpu_ms(modelo, ds["test"]),
        "tiempo_entrenamiento_s": time.time() - t0,
        "dispositivo_entrenamiento": disp,
        "historia": historia,
    }
    salida.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), salida / "model.pt")  # ya en CPU
    (salida / "metrics.json").write_text(json.dumps(metr, indent=2, ensure_ascii=False))
    (salida / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (salida / "data_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return metr


def registrar_mlflow(cfg, metr, salida: Path):
    import os

    # Windows con "Control inteligente de aplicaciones" bloquea las extensiones compiladas de SQLAlchemy
    # (_collections_cy). Esta variable hace que SQLAlchemy use su versión en Python puro.
    os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")
    try:
        import mlflow
    except Exception as e:
        print(f"  (MLflow no disponible: se omite el registro) {e}")
        return
    try:
        _registrar(mlflow, cfg, metr, salida, "sqlite:///" + (RAIZ / "mlflow.db").as_posix())
    except Exception as e1:
        # Respaldo: almacén en carpeta (no usa SQLAlchemy)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        try:
            _registrar(mlflow, cfg, metr, salida, (RAIZ / "mlruns").as_uri())
            print("  (MLflow: se usó el almacén en carpeta mlruns/ porque SQLite falló)")
        except Exception as e2:  # el registro nunca debe tumbar el entrenamiento
            print(f"  (MLflow falló, la corrida igual quedó guardada en {salida}): {e1} | {e2}")


def _registrar(mlflow, cfg, metr, salida: Path, uri: str):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("lstm_vs_transformer")
    with mlflow.start_run(run_name=f"{cfg['nombre']}_s{metr['semilla']}"):
        mlflow.log_params({"modelo": metr["modelo"], "semilla": metr["semilla"], "fuente": cfg["datos"]["fuente"],
                           "ventana": cfg["ventana"]["largo"], **{f"m_{k}": v for k, v in cfg["modelo"].get("params", {}).items()}})
        mlflow.log_metrics({"test_accuracy": metr["test"]["accuracy"], "test_f1_macro": metr["test"]["f1_macro"],
                            "parametros": metr["parametros"], "latencia_cpu_ms": metr["latencia_cpu_local"]["mediana_ms"]})
        mlflow.log_artifacts(str(salida))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()
    disp = elegir_dispositivo(args.device)
    print(f"Entrenando en: {disp}" + (f" ({torch.cuda.get_device_name(0)})" if disp == "cuda" else
          "  (sin GPU CUDA disponible para PyTorch)"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # El split de datos usa una semilla FIJA: todos los modelos se comparan sobre las mismas grabaciones.
    ds, info = preparar(cfg, semilla_split=cfg.get("semilla_split", 0))
    print(f"Datos: {info['n_grabaciones']} grabaciones | {info['n_ventanas']} ventanas")

    for s in args.seeds:
        salida = RAIZ / "runs" / cfg["nombre"] / f"seed{s}"
        metr = entrenar_una(cfg, ds, info, s, salida, disp)
        print(f"==> {cfg['nombre']} s{s}: acc {metr['test']['accuracy']:.4f} | f1 {metr['test']['f1_macro']:.4f} | "
              f"{metr['parametros']} params | {metr['latencia_cpu_local']['mediana_ms']:.2f} ms (CPU local)")
        if cfg.get("mlflow", False):
            registrar_mlflow(cfg, metr, salida)


if __name__ == "__main__":
    main()
