import pytest

from app import repo
from app.config import ABORDAGENS_EMBEDDING

A = "CLIP B/32"
TABELA = ABORDAGENS_EMBEDDING[A]["tabela"]


def indices(conn):
    return [r[0] for r in conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = %s AND indexname NOT LIKE '%%pkey'",
        (TABELA,)).fetchall()]


def test_cria_tabela_com_n_linhas_e_6_indices(imagens_db, embed_falso):
    repo.preparar_modelo(A)
    assert imagens_db and repo.modelo_preparado(A)
    from app.db import conexao
    with conexao() as c:
        assert c.execute(f"SELECT count(*) FROM {TABELA}").fetchone()[0] == 5
        defs = indices(c)
    assert len(defs) == 6
    assert sum("hnsw" in d for d in defs) == 3 and sum("ivfflat" in d for d in defs) == 3
    # vector_l2_ops é o padrão e o indexdef o omite
    assert sum("vector_cosine_ops" in d for d in defs) == 2
    assert sum("vector_ip_ops" in d for d in defs) == 2


def test_grava_config_e_tempo(imagens_db, embed_falso):
    r = repo.preparar_modelo(A)
    cfg = repo.obter_config(A)
    assert cfg["total"] == 5 and cfg["tempo_geracao_s"] > 0 and cfg["dimensao"] == 512
    assert cfg["tabela"] == TABELA
    assert r["total"] == 5 and r["puladas"] == 0


def test_progresso_chamado_por_lote(imagens_db, embed_falso, monkeypatch):
    monkeypatch.setattr(repo, "TAMANHO_LOTE", 2)
    chamadas = []
    repo.preparar_modelo(A, progresso=lambda f, t: chamadas.append((f, t)))
    assert chamadas == [(2, 5), (4, 5), (5, 5)]


def test_repreparar_nao_duplica(imagens_db, embed_falso):
    repo.preparar_modelo(A)
    repo.preparar_modelo(A)
    from app.db import conexao
    with conexao() as c:
        assert c.execute(f"SELECT count(*) FROM {TABELA}").fetchone()[0] == 5
        assert len(indices(c)) == 6


def test_imagem_ilegivel_e_pulada(imagens_db, embed_falso, monkeypatch):
    def com_falha(caminhos, abordagem):
        r = embed_falso(caminhos, abordagem)
        r[0] = None
        return r
    monkeypatch.setattr(repo, "embed_imagens", com_falha)
    monkeypatch.setattr(repo, "TAMANHO_LOTE", 10)
    r = repo.preparar_modelo(A)
    assert r["puladas"] == 1 and r["total"] == 4


def test_sem_imagens_levanta_erro_claro(banco_limpo, embed_falso):
    with pytest.raises(ValueError):
        repo.preparar_modelo(A)
    assert not repo.modelo_preparado(A)


def test_falha_no_meio_descarta_tabela(imagens_db, embed_falso, monkeypatch):
    monkeypatch.setattr(repo, "TAMANHO_LOTE", 2)
    n = {"c": 0}

    def quebra(caminhos, abordagem):
        n["c"] += 1
        if n["c"] == 2:
            raise RuntimeError("falha")
        return embed_falso(caminhos, abordagem)
    monkeypatch.setattr(repo, "embed_imagens", quebra)
    with pytest.raises(RuntimeError):
        repo.preparar_modelo(A)
    assert not repo.modelo_preparado(A)
    assert repo.obter_config(A) is None


def test_total_imagens(imagens_db):
    assert repo.total_imagens() == 5
