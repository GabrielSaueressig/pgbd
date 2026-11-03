"""Abordagens de embedding e de busca selecionáveis na interface."""

ABORDAGENS_EMBEDDING = {
    "CLIP B/32": {"modelo": "clip-ViT-B-32", "dimensao": 512},
    "CLIP L/14": {"modelo": "clip-ViT-L-14", "dimensao": 768},
    "SigLIP": {"modelo": "google/siglip-base-patch16-224", "dimensao": 768},
}

ABORDAGENS_BUSCA = [
    "Vetorial exata",
    "Vetorial HNSW",
    "Vetorial IVFFlat",
    "Trigram (pg_trgm)",
    "Full-text (tsvector)",
]

METRICAS = {
    "Cosseno": "<=>",
    "L2": "<->",
    "Produto interno": "<#>",
}

TAMANHO_LOTE = 16
PASTA_IMAGENS = "imagens"
