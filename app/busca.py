"""Busca vetorial por texto: exata, HNSW e IVFFlat, com a estratégia forçada por SET LOCAL."""

import re
import statistics
import time

import numpy as np

from app.config import ABORDAGENS_BUSCA, EF_SEARCH, METRICAS, PROBES
from app.db import conexao
from app.embeddings import embed_texto
from app.repo import obter_config

_AJUSTES = {
    "Vetorial exata": ["enable_indexscan = off", "enable_bitmapscan = off", "enable_indexonlyscan = off"],
    "Vetorial HNSW": ["enable_seqscan = off", f"hnsw.ef_search = {EF_SEARCH}"],
    "Vetorial IVFFlat": ["enable_seqscan = off", f"ivfflat.probes = {PROBES}"],
}
_NEGA = {"Cosseno": False, "L2": True, "Produto interno": True}


def _validar(texto: str, modelo_busca: str, metrica: str, abordagem: str) -> dict:
    if not texto or not texto.strip():
        raise ValueError("Digite um texto para buscar")
    if modelo_busca not in ABORDAGENS_BUSCA or metrica not in METRICAS:
        raise ValueError("Abordagem de busca ou métrica inválida")
    cfg = obter_config(abordagem)
    if cfg is None:
        raise ValueError(f"O modelo {abordagem} ainda não foi preparado")
    return cfg


def _sql(tabela: str, metrica: str) -> str:
    op = METRICAS[metrica]  # vem de config.py, nunca de entrada do usuário
    return (
        f"SELECT i.caminho, i.nome, e.vetor {op} %(q)s AS dist FROM {tabela} e "
        f"JOIN imagem i ON i.id = e.imagem_id ORDER BY e.vetor {op} %(q)s LIMIT %(k)s"
    )


def _ajustar(conn, modelo_busca: str) -> None:
    for a in _AJUSTES[modelo_busca]:
        conn.execute(f"SET LOCAL {a}")


def buscar(texto: str, modelo_busca: str, k: int, metrica: str, abordagem: str) -> dict:
    """Top-k com score, mediana de 5 execuções (1ª à parte) e o plano; embedding fora da medição."""
    cfg = _validar(texto, modelo_busca, metrica, abordagem)
    q = np.array(embed_texto(texto.strip(), abordagem), dtype=np.float32)
    sql, params = _sql(cfg["tabela"], metrica), {"q": q, "k": k}
    tempos, linhas = [], []
    with conexao() as conn, conn.transaction():
        _ajustar(conn, modelo_busca)
        for _ in range(5):
            t0 = time.perf_counter()
            linhas = conn.execute(sql, params).fetchall()
            tempos.append((time.perf_counter() - t0) * 1000)
        plano = "\n".join(r[0] for r in conn.execute("EXPLAIN (ANALYZE) " + sql, params).fetchall())
    resultados = []
    for caminho, nome, dist in linhas:
        score = -dist if _NEGA[metrica] else 1 - dist
        resultados.append({"caminho": caminho, "nome": nome, "score": float(score)})
    m = re.search(r"Index (?:Only )?Scan using (\S+)", plano)
    return {
        "resultados": resultados,
        "tempo_ms": statistics.median(tempos),
        "tempo_ms_1a": tempos[0],
        "plano": plano,
        "indice": m.group(1) if m else None,
    }


def explicar(texto: str, modelo_busca: str, k: int, metrica: str, abordagem: str) -> str:
    return buscar(texto, modelo_busca, k, metrica, abordagem)["plano"]
