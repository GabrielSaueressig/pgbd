"""Abordagens de busca: vetorial (exata, HNSW, IVFFlat), trigram e full-text."""

import time

from app.config import METRICAS
from app.db import conexao
from app.embeddings import embed_texto


def buscar(texto: str, modelo_busca: str, k: int, **params) -> list[dict]:
    """Retorna os top-k com caminho, nome, score e tempo_ms."""
    inicio = time.perf_counter()
    # Área 3: uma consulta por modelo_busca; medir tempo; scores em ordem decrescente.
    resultados: list[dict] = []
    tempo_ms = (time.perf_counter() - inicio) * 1000
    for r in resultados:
        r["tempo_ms"] = tempo_ms
    return resultados
