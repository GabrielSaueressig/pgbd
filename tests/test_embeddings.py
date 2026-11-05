import math

import pytest
from PIL import Image

from app.embeddings import aquecer, dimensao, embed_imagem, embed_imagens, embed_texto

MODELOS = [
    "CLIP B/32",
    pytest.param("CLIP L/14", marks=pytest.mark.slow),
    pytest.param("SigLIP", marks=pytest.mark.slow),
]


def norma(v):
    return math.sqrt(sum(x * x for x in v))


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.parametrize("abordagem", MODELOS)
def test_dimensao_e_norma_imagem(abordagem, imagens_tmp):
    v = embed_imagem(imagens_tmp[0], abordagem)
    assert len(v) == dimensao(abordagem)
    assert abs(norma(v) - 1.0) < 1e-4


@pytest.mark.parametrize("abordagem", MODELOS)
def test_dimensao_e_norma_texto(abordagem):
    v = embed_texto("praia", abordagem)
    assert len(v) == dimensao(abordagem)
    assert abs(norma(v) - 1.0) < 1e-4


def test_lote_devolve_none_para_imagem_corrompida(imagens_tmp, tmp_path):
    ruim = tmp_path / "ruim.jpg"
    ruim.write_text("não sou imagem")
    r = embed_imagens([imagens_tmp[0], str(ruim)], "CLIP B/32")
    assert r[0] is not None and len(r[0]) == 512
    assert r[1] is None


@pytest.mark.parametrize("modo", ["CMYK", "RGBA", "P"])
def test_modos_de_cor_convertidos(modo, tmp_path):
    p = tmp_path / ("c.jpg" if modo == "CMYK" else f"{modo}.png")
    Image.new("RGB", (32, 32), "red").convert(modo).save(p)
    v = embed_imagem(str(p), "CLIP B/32")
    assert abs(norma(v) - 1.0) < 1e-4


def test_texto_relaciona_imagem(imagens_tmp):
    vermelha, azul = imagens_tmp[0], imagens_tmp[1]
    t = embed_texto("a solid red image", "CLIP B/32")
    assert cos(t, embed_imagem(vermelha, "CLIP B/32")) > cos(t, embed_imagem(azul, "CLIP B/32"))


def test_aquecer_nao_levanta():
    aquecer("CLIP B/32")
