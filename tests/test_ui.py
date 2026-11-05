import pytest
from streamlit.testing.v1 import AppTest

from app import busca, carga, db, embeddings, repo


@pytest.fixture
def ui(monkeypatch, imagens_tmp):
    estado = {"total": 5, "preparado": True, "preparou": False}
    monkeypatch.setattr(db, "criar_schema", lambda: None)
    monkeypatch.setattr(carga, "sincronizar_pasta", lambda *a, **k: 0)
    monkeypatch.setattr(embeddings, "aquecer", lambda a: None)
    monkeypatch.setattr(repo, "total_imagens", lambda: estado["total"])
    monkeypatch.setattr(repo, "modelo_preparado", lambda a: estado["preparado"])
    monkeypatch.setattr(repo, "obter_config", lambda a: {"tempo_geracao_s": 12.5, "total": 5} if estado["preparado"] else None)

    def prep(a, progresso=None):
        if progresso:
            progresso(5, 5)
        estado["preparado"], estado["preparou"] = True, True
        return {"total": 5, "puladas": 0, "tempo_geracao_s": 1.0}

    monkeypatch.setattr(repo, "preparar_modelo", prep)

    def fake_buscar(texto, modo, k, metrica, abordagem):
        return {
            "resultados": [{"caminho": p, "nome": p.split("/")[-1], "score": 0.9 - i / 10}
                           for i, p in enumerate(imagens_tmp[:k])],
            "tempo_ms": 1.5, "tempo_ms_1a": 4.2, "plano": "Seq Scan on emb", "indice": None,
        }

    monkeypatch.setattr(busca, "buscar", fake_buscar)
    at = AppTest.from_file("app/ui.py", default_timeout=30)
    at.estado = estado
    return at


def buscar(at, texto="praia"):
    at.text_input[0].set_value(texto).run()
    at.button(key="buscar").click().run()
    return at


def test_sem_imagens_mostra_aviso(ui):
    ui.estado["total"] = 0
    ui.run()
    assert not ui.exception
    assert any("imagens" in w.value for w in ui.warning)
    assert ui.button(key="buscar").disabled


def test_modelo_nao_preparado_mostra_botao_preparar(ui):
    ui.estado["preparado"] = False
    ui.run()
    assert ui.button(key="preparar")
    assert ui.button(key="buscar").disabled


def test_preparar_modelo_libera_busca(ui):
    ui.estado["preparado"] = False
    ui.run()
    ui.button(key="preparar").click().run()
    assert ui.estado["preparou"] and not ui.exception
    assert not ui.button(key="buscar").disabled


def test_busca_mostra_3_imagens_e_tempos(ui):
    buscar(ui.run())
    assert not ui.exception
    assert len(ui.get("imgs")) == 3
    rotulos = {m.label: m.value for m in ui.metric}
    assert "Tempo de busca" in rotulos and "Tempo de gerar embeddings" in rotulos
    assert any(e.label == "EXPLAIN" for e in ui.expander)


def test_texto_vazio_nao_busca(ui):
    ui.run()
    ui.button(key="buscar").click().run()
    assert not ui.exception and len(ui.get("imgs")) == 0


def test_arquivo_ausente_nao_quebra(ui, monkeypatch):
    def sem_arquivo(texto, modo, k, metrica, abordagem):
        return {"resultados": [{"caminho": "/nao/existe.jpg", "nome": "x.jpg", "score": 0.5}] * 3,
                "tempo_ms": 1.0, "tempo_ms_1a": 1.0, "plano": "", "indice": None}
    monkeypatch.setattr(busca, "buscar", sem_arquivo)
    buscar(ui.run())
    assert not ui.exception and any("não encontrad" in w.value for w in ui.warning)


def test_erro_de_busca_vira_mensagem(ui, monkeypatch):
    def erro(*a, **k):
        raise ValueError("modelo não preparado")
    monkeypatch.setattr(busca, "buscar", erro)
    buscar(ui.run())
    assert not ui.exception and any("modelo não preparado" in e.value for e in ui.error)
