"""Lê a pasta ./imagens e insere no catálogo as imagens novas."""

from pathlib import Path

from app.db import conexao

EXTENSOES = {".jpg", ".jpeg", ".png"}


def sincronizar_pasta(raiz: str = "imagens") -> int:
    """Insere as imagens novas. Tags vêm do nome da subpasta. Retorna quantas foram inseridas."""
    base = Path(raiz)
    inseridas = 0
    with conexao() as conn:
        for arq in base.rglob("*"):
            if arq.suffix.lower() not in EXTENSOES:
                continue
            tags = [arq.parent.name] if arq.parent != base else []
            cur = conn.execute(
                """
                INSERT INTO imagem (nome, caminho, tags)
                VALUES (%s, %s, %s)
                ON CONFLICT (caminho) DO NOTHING
                """,
                (arq.name, str(arq), tags),
            )
            inseridas += cur.rowcount
    return inseridas
