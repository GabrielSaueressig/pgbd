# Busca de imagens com pgvector

Interface Streamlit para comparar abordagens de embedding (CLIP B/32, CLIP L/14, SigLIP) e de busca vetorial (exata, HNSW, IVFFlat) sobre PostgreSQL 16 + pgvector. Especificação em `especificacao-busca-imagens.md`, plano em `plano-implementacao.md`.

## Pré-requisitos
- Docker com Compose
- Python 3.12 e [uv](https://docs.astral.sh/uv/) (recomendado) ou o pacote `python3-venv`
- Imagens em `imagens/<categoria>/` (subconjunto do Unsplash 25k, ~5k imagens)

## Setup

```bash
scripts/init.sh
```

O script é idempotente e faz: `.env` a partir de `.env.example`, `docker compose up -d`, espera do Postgres, aplica `schema.sql`, cria `.venv` e instala `requirements.txt`, e pré-baixa os 3 modelos. Para pular o download dos modelos (vários GB): `scripts/init.sh --sem-modelos`.

## Executar

```bash
source .venv/bin/activate
streamlit run app/ui.py
```

## Parar o banco

```bash
docker compose down        # mantém os dados
docker compose down -v     # apaga o volume
```
