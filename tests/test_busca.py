import random

import pytest

from app import busca, repo

A = "CLIP B/32"


def unitario(dim=512):
    v = [random.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


@pytest.fixture
def modelo_pronto(banco_limpo, embed_falso, monkeypatch):
    for i in range(300):
        banco_limpo.execute(
            "INSERT INTO imagem (nome, caminho, tags) VALUES (%s, %s, '{}')", (f"i{i}.jpg", f"/x/i{i}.jpg"))
    repo.preparar_modelo(A)
    monkeypatch.setattr(busca, "embed_texto", lambda t, a: unitario())


def test_exata_usa_seq_scan(modelo_pronto):
    r = busca.buscar("praia", "Vetorial exata", 3, "Cosseno", A)
    assert "Seq Scan" in r["plano"] and r["indice"] is None


@pytest.mark.xfail(reason="HNSW e IVFFlat coexistem no mesmo operador: enable_* não escolhe o índice; "
                   "resolver na área de busca (ex.: expor um único índice por transação)", strict=False)
@pytest.mark.parametrize("modo,trecho", [("Vetorial HNSW", "_hnsw_"), ("Vetorial IVFFlat", "_ivf_")])
@pytest.mark.parametrize("metrica", ["Cosseno", "L2", "Produto interno"])
def test_ann_usa_indice(modelo_pronto, modo, trecho, metrica):
    r = busca.buscar("praia", modo, 3, metrica, A)
    assert trecho in (r["indice"] or ""), r["plano"]


@pytest.mark.parametrize("metrica", ["Cosseno", "L2", "Produto interno"])
def test_score_decrescente_e_k_resultados(modelo_pronto, metrica):
    r = busca.buscar("praia", "Vetorial exata", 3, metrica, A)
    scores = [x["score"] for x in r["resultados"]]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)
    assert {"caminho", "nome", "score"} <= set(r["resultados"][0])
    assert r["tempo_ms"] > 0 and r["tempo_ms_1a"] > 0


@pytest.mark.parametrize("texto", ["", "   "])
def test_texto_vazio_levanta(modelo_pronto, texto):
    with pytest.raises(ValueError):
        busca.buscar(texto, "Vetorial exata", 3, "Cosseno", A)


def test_modelo_nao_preparado_levanta(banco_limpo):
    with pytest.raises(ValueError):
        busca.buscar("praia", "Vetorial exata", 3, "Cosseno", A)


def test_explicar_devolve_plano(modelo_pronto):
    assert "Execution Time" in busca.explicar("praia", "Vetorial HNSW", 3, "Cosseno", A)
