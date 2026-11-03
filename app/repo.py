"""Tabelas de embedding, regeneração ao vivo e configuração ativa."""

from datetime import datetime, timezone

from app.config import ABORDAGENS_EMBEDDING, TAMANHO_LOTE
from app.db import conexao
from app.embeddings import dimensao, embed_imagem


def aplicar_abordagem(abordagem: str, progresso=None) -> None:
    """Recria a tabela embedding com a dimensão da abordagem e regenera todos os vetores."""
    modelo = ABORDAGENS_EMBEDDING[abordagem]["modelo"]
    dim = dimensao(abordagem)
    with conexao() as conn:
        conn.execute("DROP TABLE IF EXISTS embedding")
        conn.execute(
            f"""
            CREATE TABLE embedding (
              imagem_id INT PRIMARY KEY REFERENCES imagem(id) ON DELETE CASCADE,
              vetor     vector({dim}) NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX embedding_hnsw ON embedding USING hnsw (vetor vector_cosine_ops)")

        imagens = conn.execute("SELECT id, caminho FROM imagem ORDER BY id").fetchall()
        total = len(imagens)
        for inicio in range(0, total, TAMANHO_LOTE):
            for img_id, caminho in imagens[inicio:inicio + TAMANHO_LOTE]:
                vetor = embed_imagem(caminho, abordagem)
                conn.execute(
                    "INSERT INTO embedding (imagem_id, vetor) VALUES (%s, %s)",
                    (img_id, vetor),
                )
            if progresso:
                progresso(min(inicio + TAMANHO_LOTE, total), total)

        conn.execute(
            """
            INSERT INTO config_embedding (id, modelo, dimensao, gerado_em, total)
            VALUES (1, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              modelo = EXCLUDED.modelo, dimensao = EXCLUDED.dimensao,
              gerado_em = EXCLUDED.gerado_em, total = EXCLUDED.total
            """,
            (modelo, dim, datetime.now(timezone.utc), total),
        )


def obter_config() -> dict:
    with conexao() as conn:
        linha = conn.execute(
            "SELECT modelo, dimensao, gerado_em, total FROM config_embedding WHERE id = 1"
        ).fetchone()
    if linha is None:
        return {"modelo": None, "dimensao": None, "gerado_em": None, "total": 0}
    return dict(zip(["modelo", "dimensao", "gerado_em", "total"], linha))
