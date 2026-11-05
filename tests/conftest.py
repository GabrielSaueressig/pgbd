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
