import pytest
from PIL import Image


@pytest.fixture
def conn():
    try:
        from app.db import conexao
        c = conexao()
    except Exception as e:  # banco indisponível
        pytest.skip(f"banco indisponível: {e}")
    yield c
    c.close()


@pytest.fixture
def imagens_tmp(tmp_path):
    cores = ["red", "blue", "green", "yellow", "black"]
    caminhos = []
    for i, cor in enumerate(cores):
        p = tmp_path / f"{cor}.jpg"
        Image.new("RGB", (64, 64), cor).save(p)
        caminhos.append(str(p))
    return caminhos


@pytest.fixture(scope="session", autouse=True)
def banco_teste():
    """Usa um banco separado (imagens_test) para nunca tocar nos dados reais."""
    import os
    from pathlib import Path

    import psycopg
    from dotenv import load_dotenv

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        return
    base, _, _ = url.rpartition("/")
    try:
        with psycopg.connect(url, autocommit=True) as c:
            if not c.execute("SELECT 1 FROM pg_database WHERE datname='imagens_test'").fetchone():
                c.execute("CREATE DATABASE imagens_test")
        os.environ["DATABASE_URL"] = f"{base}/imagens_test"
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
            c.execute(Path("schema.sql").read_text())
    except Exception:
        os.environ.pop("DATABASE_URL", None)  # as fixtures com `conn` fazem skip


@pytest.fixture
def banco_limpo(conn):
    """Esvazia imagem, config_embedding e tabelas emb_* antes e depois do teste."""
    def limpar():
        tabelas = [r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'emb\\_%'").fetchall()]
        for t in tabelas:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute("TRUNCATE imagem, config_embedding RESTART IDENTITY CASCADE")
    limpar()
    yield conn
    limpar()


@pytest.fixture
def imagens_db(banco_limpo, imagens_tmp):
    """Insere as 5 imagens sintéticas no catálogo."""
    from pathlib import Path
    for p in imagens_tmp:
        banco_limpo.execute(
            "INSERT INTO imagem (nome, caminho, tags) VALUES (%s, %s, %s)",
            (Path(p).name, p, ["teste"]))
    return imagens_tmp


@pytest.fixture
def embed_falso(monkeypatch):
    """Troca embed_imagens por vetores unitários aleatórios (dispensa o modelo)."""
    import random

    from app import repo
    from app.config import ABORDAGENS_EMBEDDING

    def falso(caminhos, abordagem):
        dim = ABORDAGENS_EMBEDDING[abordagem]["dimensao"]
        out = []
        for _ in caminhos:
            v = [random.gauss(0, 1) for _ in range(dim)]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out

    monkeypatch.setattr(repo, "embed_imagens", falso)
    return falso
