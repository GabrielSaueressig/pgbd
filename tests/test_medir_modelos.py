import math

from app.config import ABORDAGENS_EMBEDDING
from scripts import medir_modelos


def _unit(dim):
    return [1 / math.sqrt(dim)] * dim


def _falso(dim, nones=()):
    def f(caminhos, abordagem):
        return [None if i in nones else _unit(dim) for i in range(len(caminhos))]
    return f


def _prep(monkeypatch, falso):
    monkeypatch.setattr(medir_modelos, "embed_imagens", falso)
    monkeypatch.setattr(medir_modelos, "aquecer", lambda a: None)


def test_medir_calcula_campos(monkeypatch):
    dim = ABORDAGENS_EMBEDDING["CLIP B/32"]["dimensao"]
    _prep(monkeypatch, _falso(dim))
    r = medir_modelos.medir("CLIP B/32", ["a"] * 5)
    assert r["imagens"] == 5 and r["falhas"] == 0
    assert r["dimensao_ok"] and r["norma_ok"] and r["ms_por_imagem"] >= 0


def test_medir_conta_falhas_sem_abortar(monkeypatch):
    dim = ABORDAGENS_EMBEDDING["CLIP B/32"]["dimensao"]
    _prep(monkeypatch, _falso(dim, nones={1, 3}))
    r = medir_modelos.medir("CLIP B/32", ["a"] * 5)
    assert r["falhas"] == 2 and r["imagens"] == 5
    assert r["norma_ok"]


def test_medir_detecta_dimensao_errada(monkeypatch):
    _prep(monkeypatch, _falso(7))
    r = medir_modelos.medir("CLIP B/32", ["a"] * 3)
    assert not r["dimensao_ok"]
