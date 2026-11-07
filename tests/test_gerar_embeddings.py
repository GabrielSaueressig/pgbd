import pytest

from app import repo
from scripts import gerar_embeddings
from scripts.gerar_embeddings import gerar


@pytest.fixture
def sem_sincronizar(monkeypatch):
    """As imagens já estão no catálogo (imagens_db); não varre pasta."""
    monkeypatch.setattr(gerar_embeddings, "sincronizar_pasta", lambda pasta: 0)


def test_gera_modelo_e_registra_config(imagens_db, embed_falso, sem_sincronizar):
    res = gerar(["CLIP B/32"])
    assert isinstance(res["CLIP B/32"], dict)
    assert repo.obter_config("CLIP B/32")["total"] == 5


def test_pula_modelo_ja_preparado(imagens_db, embed_falso, sem_sincronizar, monkeypatch):
    chamadas = []
    original = gerar_embeddings.preparar_modelo

    def contador(abordagem, progresso=None):
        chamadas.append(abordagem)
        return original(abordagem, progresso)

    monkeypatch.setattr(gerar_embeddings, "preparar_modelo", contador)
    gerar(["CLIP B/32"])
    gerar(["CLIP B/32"])
    assert chamadas == ["CLIP B/32"]
    gerar(["CLIP B/32"], refazer=True)
    assert chamadas == ["CLIP B/32", "CLIP B/32"]


def test_falha_de_um_modelo_nao_impede_os_outros(imagens_db, embed_falso, sem_sincronizar, monkeypatch):
    original = gerar_embeddings.preparar_modelo

    def falso(abordagem, progresso=None):
        if abordagem == "SigLIP":
            raise RuntimeError("sem sentencepiece")
        return original(abordagem, progresso)

    monkeypatch.setattr(gerar_embeddings, "preparar_modelo", falso)
    res = gerar(["SigLIP", "CLIP B/32"])
    assert res["SigLIP"] == "sem sentencepiece"
    assert isinstance(res["CLIP B/32"], dict)


def test_sem_imagens_no_catalogo_da_erro_claro(banco_limpo, tmp_path):
    res = gerar(["CLIP B/32"], pasta=str(tmp_path))
    assert "Não há imagens no catálogo" in res["CLIP B/32"]
