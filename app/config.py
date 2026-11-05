"""Abordagens de embedding e de busca selecionáveis na interface."""

ABORDAGENS_EMBEDDING = {
    "CLIP B/32": {"modelo": "openai/clip-vit-base-patch32", "dimensao": 512, "tabela": "emb_clip_b32"},
    "CLIP L/14": {"modelo": "openai/clip-vit-large-patch14", "dimensao": 768, "tabela": "emb_clip_l14"},
    "SigLIP": {"modelo": "google/siglip-base-patch16-224", "dimensao": 768, "tabela": "emb_siglip"},
}

ABORDAGENS_BUSCA = ["Vetorial exata", "Vetorial HNSW", "Vetorial IVFFlat"]

METRICAS = {
    "Cosseno": "<=>",
    "L2": "<->",
    "Produto interno": "<#>",
}

OPCLASS_POR_METRICA = {
    "Cosseno": "vector_cosine_ops",
    "L2": "vector_l2_ops",
    "Produto interno": "vector_ip_ops",
}

EF_SEARCH = 40
PROBES = 10
K_FIXO = 3
TAMANHO_LOTE = 16
PASTA_IMAGENS = "imagens"
