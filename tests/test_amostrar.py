from PIL import Image

from scripts.amostrar_imagens import amostrar


def _criar(pasta, nomes):
    pasta.mkdir(parents=True, exist_ok=True)
    for n in nomes:
        Image.new("RGB", (8, 8), "red").save(pasta / n)


def _origem(tmp_path, n=10):
    origem = tmp_path / "origem"
    _criar(origem, [f"f{i}.jpg" for i in range(n)])
    return origem


def test_copia_n_imagens_na_categoria(tmp_path):
    origem = _origem(tmp_path)
    (origem / "fotos.csv").write_text("a,b")
    destino = tmp_path / "destino"
    assert amostrar(origem, destino, 4) == 4
    arquivos = list((destino / "unsplash").iterdir())
    assert len(arquivos) == 4
    assert all(a.suffix != ".csv" for a in arquivos)


def test_menos_imagens_que_n_copia_todas(tmp_path):
    assert amostrar(_origem(tmp_path, 3), tmp_path / "d", 10) == 3


def test_mesma_semente_mesma_amostra(tmp_path):
    origem = _origem(tmp_path)
    amostrar(origem, tmp_path / "d1", 4)
    amostrar(origem, tmp_path / "d2", 4)
    nomes = lambda d: sorted(p.name for p in (d / "unsplash").iterdir())
    assert nomes(tmp_path / "d1") == nomes(tmp_path / "d2")


def test_reexecutar_nao_duplica(tmp_path):
    origem = _origem(tmp_path)
    destino = tmp_path / "d"
    assert amostrar(origem, destino, 4) == 4
    assert amostrar(origem, destino, 4) == 0
    assert len(list((destino / "unsplash").iterdir())) == 4


def test_nomes_repetidos_em_subpastas_nao_colidem(tmp_path):
    origem = tmp_path / "origem"
    _criar(origem / "a", ["x.jpg"])
    _criar(origem / "b", ["x.jpg"])
    destino = tmp_path / "d"
    assert amostrar(origem, destino, 2) == 2
    assert len(list((destino / "unsplash").iterdir())) == 2
