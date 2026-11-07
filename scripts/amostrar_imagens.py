"""Amostra N imagens de uma pasta de origem (ex.: Unsplash 25k) para imagens/<categoria>/."""

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXTENSOES = {".jpg", ".jpeg", ".png"}


def amostrar(origem: Path, destino: Path, n: int, categoria: str = "unsplash", semente: int = 42) -> int:
    """Copia até n imagens sorteadas (determinístico pela semente); devolve quantas foram copiadas."""
    origem, destino = Path(origem), Path(destino)
    candidatas = sorted(p for p in origem.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES)
    if len(candidatas) < n:
        print(f"Aviso: só há {len(candidatas)} imagens em {origem} (pedidas: {n}); copiando todas.")
    random.Random(semente).shuffle(candidatas)
    escolhidas = candidatas[:n]

    repetidos = {nome for nome, c in Counter(p.name for p in escolhidas).items() if c > 1}
    pasta = destino / categoria
    pasta.mkdir(parents=True, exist_ok=True)
    copiadas = 0
    for p in escolhidas:
        nome = p.name
        if nome in repetidos:  # prefixa a subpasta para não sobrescrever
            rel = p.parent.relative_to(origem).as_posix().replace("/", "_")
            nome = f"{rel}_{nome}" if rel else nome
        alvo = pasta / nome
        if alvo.exists():
            continue
        shutil.copy2(p, alvo)
        copiadas += 1
    print(f"{copiadas} imagens copiadas para {pasta}")
    return copiadas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("origem", type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--categoria", default="unsplash")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--destino", type=Path, default=Path("imagens"))
    a = ap.parse_args()
    amostrar(a.origem, a.destino, a.n, a.categoria, a.semente)


if __name__ == "__main__":
    main()
