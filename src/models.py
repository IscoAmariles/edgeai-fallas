"""
Modelos a comparar: LSTM vs. Transformer (y CNN-1D como línea base opcional).

Diseño para una comparación JUSTA (esto es lo que se justifica en la tesis):
todos los modelos comparten la misma entrada, el mismo "embebido por parches" y la misma
cabeza de fusión con temperatura y diferencial de presión. Lo ÚNICO que cambia es el
codificador de la secuencia. Así, la diferencia de métricas se puede atribuir a la arquitectura.

  vib (B,3,L) -> PatchEmbed -> (B,T,d) -> [LSTM | Transformer | CNN] -> (B,d) --+
  aux (B,2,La) -> estadísticos (media, std, pendiente) -> MLP -> (B,d_aux) -----+-> clasificador

Parchear la señal (p. ej. 1024 muestras en 32 parches de 32) reduce la secuencia de 1024 a 32
pasos: crítico para que la atención del Transformer quepa en ≤50 ms en la CPU de la Raspberry.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, canales: int, parche: int, d: int):
        super().__init__()
        self.proj = nn.Conv1d(canales, d, kernel_size=parche, stride=parche)

    def forward(self, x):  # (B,C,L) -> (B,T,d)
        return self.proj(x).transpose(1, 2)


class CodLSTM(nn.Module):
    def __init__(self, d: int, capas: int = 1, dropout: float = 0.1, **_):
        super().__init__()
        self.lstm = nn.LSTM(d, d, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)

    def forward(self, h):
        out, _ = self.lstm(h)
        return out[:, -1]  # último estado


class CodTransformer(nn.Module):
    def __init__(self, d: int, n_tokens: int, capas: int = 2, cabezas: int = 4, ff: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        capa = nn.TransformerEncoderLayer(d, cabezas, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        h = self.enc(h + self.pos)
        return self.norm(h.mean(dim=1))  # promedio sobre tokens


class CodCNN(nn.Module):
    def __init__(self, d: int, capas: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        bloques = []
        for _ in range(capas):
            bloques += [nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*bloques)

    def forward(self, h):
        return self.net(h.transpose(1, 2)).mean(dim=2)


class ClasificadorFallas(nn.Module):
    def __init__(self, tipo: str, forma_vib, forma_aux, n_clases: int, d: int = 64, parche: int = 32, d_aux: int = 16, **kw):
        super().__init__()
        c, L = forma_vib
        assert L % parche == 0, "El largo de ventana debe ser múltiplo del parche"
        n_tok = L // parche
        self.embed = PatchEmbed(c, parche, d)
        if tipo == "lstm":
            self.cod = CodLSTM(d, **kw)
        elif tipo == "transformer":
            self.cod = CodTransformer(d, n_tok, **kw)
        elif tipo == "cnn":
            self.cod = CodCNN(d, **kw)
        else:
            raise ValueError(tipo)
        ca = forma_aux[0]
        self.aux = nn.Sequential(nn.Linear(3 * ca, d_aux), nn.ReLU())
        self.cabeza = nn.Sequential(nn.Linear(d + d_aux, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, n_clases))

    @staticmethod
    def _estad_aux(a):  # (B,C,La) -> (B,3C): media, desviación, pendiente
        La = a.shape[-1]
        t = torch.linspace(-1, 1, La, device=a.device)
        media = a.mean(-1)
        std = ((a - media.unsqueeze(-1)) ** 2).mean(-1).sqrt()
        pend = (a * t).mean(-1)
        return torch.cat([media, std, pend], dim=1)

    def forward(self, vib, aux):
        h = self.cod(self.embed(vib))
        z = self.aux(self._estad_aux(aux))
        return self.cabeza(torch.cat([h, z], dim=1))


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
