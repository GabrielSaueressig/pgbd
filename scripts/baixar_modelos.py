"""Pré-baixa os modelos de embedding para o cache local (Hugging Face)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ABORDAGENS_EMBEDDING


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor

    for nome, cfg in ABORDAGENS_EMBEDDING.items():
        modelo = cfg["modelo"]
        print(f"Baixando {nome} ({modelo})...", flush=True)
        if modelo.startswith("clip-"):
            SentenceTransformer(modelo, device="cpu")
        else:
            AutoProcessor.from_pretrained(modelo)
            AutoModel.from_pretrained(modelo)
    print("Modelos em cache.")


if __name__ == "__main__":
    main()
