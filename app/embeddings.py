"""Carregamento dos modelos e geração dos vetores de imagem e de texto."""

from functools import lru_cache

from app.config import ABORDAGENS_EMBEDDING


@lru_cache(maxsize=3)
def carregar_modelo(abordagem: str):
    """Carrega o modelo da abordagem uma vez e mantém em cache."""
    raise NotImplementedError("Área 2: implementar com sentence-transformers / transformers")


def embed_imagem(caminho: str, abordagem: str) -> list[float]:
    raise NotImplementedError("Área 2")


def embed_texto(texto: str, abordagem: str) -> list[float]:
    raise NotImplementedError("Área 2")


def dimensao(abordagem: str) -> int:
    return ABORDAGENS_EMBEDDING[abordagem]["dimensao"]
