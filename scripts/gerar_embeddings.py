"""Gera os embeddings (e os 6 índices) de todos os modelos sobre a pasta de imagens."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.carga import sincronizar_pasta
from app.config import PASTA_IMAGENS
from app.repo import obter_config, preparar_modelo

ORDEM_PADRAO = ["CLIP B/32", "SigLIP", "CLIP L/14"]


def gerar(modelos: list[str], pasta: str = PASTA_IMAGENS, refazer: bool = False) -> dict[str, dict | str]:
    """Devolve, por modelo, o resultado de preparar_modelo ou a mensagem de erro."""
    novas = sincronizar_pasta(pasta)
    print(f"Catálogo sincronizado ({novas} imagens novas).", flush=True)
    resultados: dict[str, dict | str] = {}
    for nome in modelos:
        if not refazer and obter_config(nome) is not None:
            print(f"[{nome}] já preparado, pulando.", flush=True)
            resultados[nome] = obter_config(nome)
            continue

        def progresso(feito: int, total: int, nome=nome) -> None:
            print(f"\r[{nome}] {feito}/{total}", end="", flush=True)

        try:
            resultados[nome] = preparar_modelo(nome, progresso)
            r = resultados[nome]
            print(f"\n[{nome}] ok: {r['total']} vetores em {r['tempo_geracao_s']:.0f}s", flush=True)
        except Exception as e:
            resultados[nome] = str(e)
            print(f"\n[{nome}] FALHOU: {e}", flush=True)
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelos", nargs="+", default=ORDEM_PADRAO)
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--pasta", default=PASTA_IMAGENS)
    a = ap.parse_args()
    res = gerar(a.modelos, a.pasta, a.refazer)
    if any(isinstance(v, str) for v in res.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
