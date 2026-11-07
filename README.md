# Busca de imagens com pgvector

Interface Streamlit para comparar abordagens de embedding (CLIP B/32, CLIP L/14, SigLIP) e de busca vetorial (exata, HNSW, IVFFlat) sobre PostgreSQL 16 + pgvector. Especificação em `especificacao-busca-imagens.md`, plano em `plano-implementacao.md`.

## Pré-requisitos
- Docker com Compose
- Python 3.12 e [uv](https://docs.astral.sh/uv/) (recomendado) ou o pacote `python3-venv`
- Imagens em `imagens/<categoria>/` (subconjunto do Unsplash 25k; o trabalho usa as 73 já amostradas em `imagens/unsplash/`)

> Com só 73 imagens, a busca exata e os índices HNSW/IVFFlat ficam com tempos e resultados quase idênticos (recall ~100%). A comparação de índices é limitada a esse tamanho; para uma curva mais informativa, amostre mais imagens (seção "Imagens").

## Início rápido

```bash
scripts/rodar.sh
```

Faz tudo: extrai as imagens de `dados/imagens-unsplash.zip` (se `imagens/` estiver vazia), roda o `init.sh`, gera os embeddings dos 3 modelos (pula os já prontos) e abre a interface. Opções: `--sem-modelos` (não pré-baixa os modelos), `--sem-ui` (não abre o Streamlit). O passo a passo manual está abaixo.

## Setup

```bash
scripts/init.sh
```

O script é idempotente e faz: `.env` a partir de `.env.example`, `docker compose up -d`, espera do Postgres, aplica `schema.sql`, cria `.venv` e instala `requirements.txt`, e pré-baixa os 3 modelos. Para pular o download dos modelos (vários GB): `scripts/init.sh --sem-modelos`.

## Imagens

Baixe o [Unsplash 25k](https://www.kaggle.com/datasets/ntsv648/unsplash-25k) do Kaggle (precisa de conta) para fora do repositório e amostre ~500 imagens:

```bash
python scripts/amostrar_imagens.py ~/Downloads/unsplash-25k --n 500
```

As imagens vão para `imagens/unsplash/` (ignoradas pelo git); a subpasta vira a tag.

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
