"""Mede ms/imagem de cada modelo de embedding em um subconjunto e estima o tempo da base completa."""

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ABORDAGENS_EMBEDDING, PASTA_IMAGENS, TAMANHO_LOTE
from app.embeddings import aquecer, embed_imagens

EXTENSOES = {".jpg", ".jpeg", ".png"}


def medir(abordagem: str, caminhos: list[str]) -> dict:
    dim = ABORDAGENS_EMBEDDING[abordagem]["dimensao"]
    aquecer(abordagem)  # carrega o modelo fora da cronometragem
    vetores: list = []
    inicio = time.perf_counter()
    for ini in range(0, len(caminhos), TAMANHO_LOTE):
        vetores += embed_imagens(caminhos[ini:ini + TAMANHO_LOTE], abordagem)
    tempo = time.perf_counter() - inicio
    validos = [v for v in vetores if v is not None]
    return {
        "abordagem": abordagem,
        "imagens": len(caminhos),
        "falhas": len(vetores) - len(validos),
        "ms_por_imagem": 1000 * tempo / len(caminhos) if caminhos else 0.0,
        "dimensao_ok": all(len(v) == dim for v in validos),
        "norma_ok": all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-3 for v in validos),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--modelos", nargs="+", default=list(ABORDAGENS_EMBEDDING))
    ap.add_argument("--pasta", default=PASTA_IMAGENS)
    a = ap.parse_args()

    todas = sorted(str(p) for p in Path(a.pasta).rglob("*") if p.suffix.lower() in EXTENSOES)
    if not todas:
        sys.exit(f"Nenhuma imagem em {a.pasta}")
    amostra = random.Random(42).sample(todas, min(a.n, len(todas)))

    print(f"{'modelo':<10} {'imgs':>5} {'falhas':>6} {'ms/img':>8} {'dim':>4} {'norma':>5} {'base completa (min)':>20}")
    for nome in a.modelos:
        r = medir(nome, amostra)
        est = r["ms_por_imagem"] * len(todas) / 60000
        print(f"{nome:<10} {r['imagens']:>5} {r['falhas']:>6} {r['ms_por_imagem']:>8.1f} "
              f"{'ok' if r['dimensao_ok'] else 'ERRO':>4} {'ok' if r['norma_ok'] else 'ERRO':>5} {est:>20.1f}")


if __name__ == "__main__":
    main()
