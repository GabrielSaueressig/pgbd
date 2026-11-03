# Especificação: Exploração de Embeddings e Busca Vetorial para Imagens com pgvector

Projeto final de Projeto e Gerência de Bancos de Dados. Tema: bancos vetoriais (pgvector) e abordagens de RAG/recuperação. Trabalho acadêmico: o foco é comparar abordagens, não construir um sistema de uso real. Todo o ambiente roda localmente.

**Versão 4.** Backend em Python, interface em Streamlit, sem upload. As imagens vêm de uma pasta fixa.

## 0. Mudanças da versão 4 (prevalecem sobre as seções 2, 6, 7, 8, 9 e 11)

Decididas na revisão da interface, cujo foco é a medição.

**Tecnologia e ambiente de grupo**
- Banco: **PostgreSQL 16 com a extensão pgvector** (imagem `pgvector/pgvector:pg16`), mais `pg_trgm`. Toda a busca vetorial é feita no banco, sem biblioteca vetorial externa.
- Por ser trabalho em grupo, o repositório traz um **script de inicialização** (`scripts/init.sh`) que deixa qualquer integrante com o mesmo ambiente: cria o `.env` a partir do `.env.example`, sobe o banco com `docker compose up -d`, espera o Postgres aceitar conexões, cria o venv e instala o `requirements.txt`, cria as extensões e o schema (`schema.sql`, idempotente), e pré-baixa os modelos para o cache local. Pode ser executado várias vezes sem efeito colateral.
- O script termina com o comando para iniciar a interface (`streamlit run app/ui.py`). O passo a passo também vai no README.

**Dados**
- Base: subconjunto de ~5k imagens do [Unsplash 25k](https://www.kaggle.com/datasets/ntsv648/unsplash-25k) em `./imagens`. Tags e busca textual ficam suspensas até verificar se o dataset traz metadados.
- Sem GPU na máquina de desenvolvimento: a geração roda na CPU, por isso a base de ~5k.

**Armazenamento**
- Uma tabela de embeddings **por modelo** (`emb_clip_b32`, `emb_clip_l14`, `emb_siglip`), gerada uma vez. Trocar o select de embedding apenas troca a tabela consultada.
- Cada tabela tem 6 índices, criados ao final da geração: HNSW e IVFFlat para cada uma das 3 operator classes (`vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`). O IVFFlat só é criado depois de a tabela estar populada.
- `config_embedding` passa a ter uma linha por modelo, com `tempo_geracao_s`, `total` e `gerado_em`.
- Modelo sem tabela: a UI mostra "Preparar modelo", que gera com barra de progresso, guarda o tempo total e libera a busca.

**Interface (sem sidebar)**
- Linha de controles: abordagem de embedding, abordagem de busca (Vetorial exata, Vetorial HNSW, Vetorial IVFFlat), métrica (cosseno, L2, produto interno) e campo de texto. k fixo em 3.
- Resultado: até 3 imagens com score, **tempo de busca**, **tempo de gerar embeddings** (tempo total da última geração da base para o modelo escolhido) e, num expander, o `EXPLAIN` com o índice usado.
- Parâmetros dos índices (`ef_search`, `probes`, `lists`) ficam fixos na UI e variam apenas no benchmark.
- Trigram e Full-text saem da UI; ficam no benchmark se houver metadados.

**Medição**
- Tempo de busca = só a query SQL, com o modelo já carregado (aquecimento ao selecionar). Cada busca roda 5 vezes; a UI mostra a mediana e a 1ª execução à parte.
- A estratégia é forçada por `SET LOCAL` (`enable_indexscan`/`enable_seqscan`) com os índices sempre presentes; o `EXPLAIN` confirma o plano. Não há `DROP INDEX` por consulta.
- Qualidade no benchmark: **recall do ANN (HNSW, IVFFlat) contra a busca exata**, mais tempo e tamanho de índice, variando `ef_search`, `probes` e `lists`. Gabarito manual é opcional.

## 1. Objetivo

Explorar, na prática, diferentes abordagens de embedding e de busca vetorial para recuperar imagens a partir de texto, medindo qualidade, tempo e custo de cada uma.

Na interface, o usuário escolhe:
- a **abordagem de embedding** (select); os vetores são gerados ao vivo para todo o banco;
- a **abordagem de busca** (select), a cada consulta.

## 2. Premissas e regras de negócio

- **Fonte das imagens:** pasta fixa `./imagens`. Ao iniciar a aplicação, o sistema lê a pasta e insere no banco as imagens novas. Não há upload.
- **Tags:** derivadas do nome da subpasta. Exemplo: `imagens/praia/foto1.jpg` gera a tag `praia`.
- **Troca de abordagem de embedding:** a tabela `embedding` anterior é apagada e recriada com a dimensão da nova abordagem. Em seguida, todas as imagens são reprocessadas, com barra de progresso.
- **Uma abordagem de embedding ativa por vez.** Imagem e texto usam sempre o mesmo modelo, porque vetores de modelos diferentes não são comparáveis.
- **Busca:** a abordagem de busca é escolhida a cada consulta. As buscas vetoriais exigem a abordagem de embedding ativa.

## 3. Escopo

**Incluso**
- Carga da pasta fixa de imagens no banco
- Geração ao vivo de embeddings, com seleção da abordagem e limpeza da tabela anterior
- Busca por texto com seleção da abordagem de busca
- Comparação de abordagens de embedding e de busca, com medição de qualidade e tempo

**Fora do escopo**
- Upload de imagens pela interface
- Autenticação, usuários e deploy
- Etapa de geração de texto com LLM (a etapa RAG aqui é a recuperação; a geração fica para decisão posterior do grupo)
- Treinamento de modelos

## 4. Arquitetura

```
┌───────────────────── Máquina local ─────────────────────┐
│                                                         │
│  Pasta ./imagens  ──(na inicialização)──┐               │
│                                         ▼               │
│  Streamlit (página única)                               │
│   ├─ select: abordagem de embedding  → "Gerar embeddings"│
│   └─ select: abordagem de busca + texto → resultados    │
│            │                                            │
│            ▼                                            │
│  Python (camada de serviço)                             │
│   ├─ embeddings.py : carregamento e geração dos vetores │
│   ├─ repo.py       : tabelas, carga, regeneração        │
│   └─ busca.py      : abordagens de busca                │
│            │                                            │
│            ▼                                            │
│  PostgreSQL 16 + pgvector + pg_trgm (Docker)            │
│   ├─ imagem           (catálogo, fixo)                  │
│   ├─ config_embedding (abordagem ativa)                 │
│   └─ embedding        (recriada a cada troca)           │
└─────────────────────────────────────────────────────────┘
```

## 5. Componentes e ambiente

| Componente | Tecnologia | Função |
|---|---|---|
| Banco | PostgreSQL 16 + pgvector + pg_trgm (`pgvector/pgvector:pg16`) | Vetores, índices, busca textual |
| Interface | Streamlit (página única) | Selects de abordagem, texto de busca, resultados |
| Embeddings | sentence-transformers (CLIP) e transformers (SigLIP) | Vetores de imagem e de texto |
| Acesso ao banco | psycopg 3 + pgvector-python | Consultas parametrizadas |
| Ambiente | Python 3.11, venv, `docker compose` só para o banco | Execução local |

Estrutura de pastas:

```
pgbd/final/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── especificacao-busca-imagens.md
├── benchmark.py
├── imagens/                 # pasta fixa; subpastas viram tags
│   ├── praia/
│   └── cachorro/
└── app/
    ├── config.py            # abordagens de embedding e de busca
    ├── db.py                # conexão e extensões
    ├── carga.py             # lê ./imagens e insere no catálogo
    ├── embeddings.py        # carregamento e geração dos vetores
    ├── repo.py              # tabelas, regeneração, índices
    ├── busca.py             # abordagens de busca
    └── ui.py                # Streamlit
```

## 6. Modelo de dados

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Catálogo de imagens: preenchido pela carga da pasta
CREATE TABLE imagem (
  id         SERIAL PRIMARY KEY,
  nome       TEXT NOT NULL,
  caminho    TEXT NOT NULL UNIQUE,
  tags       TEXT[] DEFAULT '{}',
  tsv        tsvector GENERATED ALWAYS AS (
               to_tsvector('simple', coalesce(nome,'') || ' ' || array_to_string(tags,' '))
             ) STORED
);
CREATE INDEX imagem_tsv_gin ON imagem USING gin (tsv);
CREATE INDEX imagem_trgm_gin ON imagem USING gin ((nome || ' ' || array_to_string(tags,' ')) gin_trgm_ops);

-- Abordagem de embedding ativa: uma linha
CREATE TABLE config_embedding (
  id         INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  modelo     TEXT NOT NULL,
  dimensao   INT NOT NULL,
  gerado_em  TIMESTAMPTZ,
  total      INT DEFAULT 0
);
```

A tabela `embedding` **não é criada no schema inicial**. A aplicação a recria a cada troca de abordagem, porque a dimensão muda:

```sql
-- executado pelo repo.py ao trocar de abordagem (dimensao = 512, 768, ...)
DROP TABLE IF EXISTS embedding;
CREATE TABLE embedding (
  imagem_id  INT PRIMARY KEY REFERENCES imagem(id) ON DELETE CASCADE,
  vetor      vector(:dimensao) NOT NULL
);
CREATE INDEX embedding_hnsw ON embedding USING hnsw (vetor vector_cosine_ops);
```

O índice HNSW é criado junto com a tabela, para permitir a comparação. Para a busca exata, o índice pode ser removido com `DROP INDEX` antes da consulta.

## 7. Abordagens de embedding

Selecionadas no select "Abordagem de embedding" da interface.

| Abordagem | Modelo | Dimensão | Observação |
|---|---|---|---|
| CLIP B/32 | `clip-ViT-B-32` | 512 | Padrão, mais rápido |
| CLIP L/14 | `clip-ViT-L-14` | 768 | Mais preciso, mais lento |
| SigLIP | `google/siglip-base-patch16-224` | 768 | Alternativa de comparação |

**Fluxo ao clicar em "Gerar embeddings":**
1. `repo.py` executa `DROP` e `CREATE` da tabela `embedding` com a nova dimensão.
2. `embeddings.py` carrega o modelo (com cache em memória).
3. Para cada imagem do catálogo, em lotes de 16: lê o arquivo, gera o vetor e insere. A barra de progresso mostra o avanço.
4. Atualiza `config_embedding` com modelo, dimensão, data e total.

## 8. Abordagens de busca

Selecionadas no select "Abordagem de busca" a cada consulta.

| Abordagem de busca | Como funciona | Parâmetros |
|---|---|---|
| Vetorial exata | `ORDER BY vetor <=> :q`, sem índice | métrica |
| Vetorial HNSW | mesma consulta, com índice HNSW | métrica, `ef_search` |
| Vetorial IVFFlat | mesma consulta, com índice IVFFlat | métrica, `lists`, `probes` |
| Trigram (pg_trgm) | similaridade de texto sobre nome e tags | limiar |
| Full-text (tsvector) | busca textual sobre nome e tags | idioma `simple` |

Métrica vetorial (select): cosseno (`<=>`), L2 (`<->`) ou produto interno (`<#>`). Cada métrica exige a operator class correspondente no índice (`vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`).

**Consulta vetorial (modelo de referência):**

```sql
SELECT i.caminho, i.nome, 1 - (e.vetor <=> %(q)s) AS score
FROM embedding e
JOIN imagem i ON i.id = e.imagem_id
ORDER BY e.vetor <=> %(q)s
LIMIT %(k)s;
```

A interface mostra, para cada consulta: os top-k resultados, o score, o tempo de execução (`time.perf_counter`) e, quando solicitado, o `EXPLAIN ANALYZE`.

## 9. Avaliação

- **Base:** 300 a 500 imagens com licença livre (ex.: subconjunto do COCO), organizadas em subpastas temáticas dentro de `./imagens`.
- **Gabarito:** 20 a 30 consultas em texto, cada uma com as imagens relevantes marcadas à mão em `gabarito.json`.
- **Métricas:**
  - Recall@5 e MRR
  - Latência média e p95 da consulta (ms)
  - Tempo de geração dos embeddings por abordagem (ms por imagem)
  - Tamanho do índice (`pg_relation_size`)
- **Entregável:** `benchmark.py`, que percorre as combinações de abordagem de embedding × abordagem de busca e gera a tabela comparativa.

## 10. Requisitos

**Funcionais**
- RF1: carga da pasta `./imagens` no catálogo, sem duplicar imagens já existentes
- RF2: select de abordagem de embedding, com geração ao vivo e limpeza da tabela anterior
- RF3: select de abordagem de busca, com parâmetros e métrica
- RF4: busca por texto com exibição de score e tempo de execução
- RF5: exibição do total de imagens e de embeddings gerados

**Não funcionais**
- RNF1: ambiente sobe com `scripts/init.sh` (banco, dependências, schema) e `streamlit run app/ui.py`; o script é idempotente (ver seção 0)
- RNF2: a interface mostra progresso durante a geração, por lote
- RNF3: senhas em `.env`, fora do repositório
- RNF4: `requirements.txt` com versões fixadas

## 11. Interface (Streamlit, página única)

**Barra lateral**
- Select "Abordagem de embedding" (CLIP B/32, CLIP L/14, SigLIP)
- Botão "Gerar embeddings" (apaga a tabela anterior e regenera todos os vetores)
- Informações: abordagem ativa, dimensão, total de imagens e de embeddings

**Área principal**
- Campo de texto e número de resultados (k)
- Select "Abordagem de busca" e parâmetros correspondentes (métrica, `ef_search`, `probes`, limiar)
- Grade de imagens com score e tempo de execução

## 12. Divisão de responsabilidades

| Área | Responsável | Entregas |
|---|---|---|
| 1. Teoria e avaliação | Pessoa A | Slides conceituais (embeddings multimodais, métricas, índices, comparação com full-text e trigram), base e gabarito, `benchmark.py`, tabela comparativa |
| 2. Backend e embeddings | Pessoa B | `db.py`, `carga.py`, `embeddings.py`, `repo.py`, schema, regeneração ao vivo, `docker-compose.yml` |
| 3. Interface e busca | Pessoa C | `busca.py`, `ui.py` (Streamlit), capturas de tela, roteiro da demo, plano B |

**Contrato entre áreas (definir antes de codar):**

```python
# carga.py
def sincronizar_pasta(raiz: str = "imagens") -> int: ...   # retorna imagens novas inseridas

# embeddings.py
def embed_imagem(caminho: str, abordagem: str) -> list[float]: ...
def embed_texto(texto: str, abordagem: str) -> list[float]: ...

# repo.py
def aplicar_abordagem(abordagem: str, progresso=None) -> None: ...   # recria a tabela e regenera
def obter_config() -> dict: ...                                       # modelo, dimensão, total

# busca.py
def buscar(texto: str, modelo_busca: str, k: int, **params) -> list[dict]: ...
# cada item: {"caminho", "nome", "score", "tempo_ms"}
```

## 13. Entregáveis

- Repositório com `docker-compose.yml`, `requirements.txt`, pasta `app/`, schema e `benchmark.py`
- `gabarito.json` e tabela comparativa de resultados
- Slides da apresentação (conceitual, técnico e demonstração), com slides reservados para as capturas de tela
- Entrega dos slides via Moodle na data definida

## 14. Critérios de aceitação

- Ao iniciar, a carga da pasta insere todas as imagens, sem duplicatas
- Trocar a abordagem de embedding remove a tabela anterior, recria com a nova dimensão e regenera todos os vetores
- Busca retorna top-k com scores em ordem decrescente, para cada abordagem de busca
- Comparação executada para pelo menos 2 abordagens de embedding, 2 métricas e 2 índices, com resultados medidos
- Demonstração ao vivo a partir de `docker compose up -d` e `streamlit run`

## 15. Riscos e plano B

| Risco | Mitigação |
|---|---|
| Download dos modelos falha na apresentação | Baixar os três modelos antes e deixá-los no cache local |
| Regeneração demora na demo | Usar subconjunto pequeno (ex.: 100 imagens) na demo e a base completa no benchmark |
| Máquina sem GPU fica lenta com CLIP L/14 e SigLIP | Usar CPU com lotes pequenos; deixar L/14 e SigLIP como comparação opcional |
| Rede ou sistema cai durante a demo | Vídeo gravado e capturas de tela preparadas |
| Resultados ruins para algumas consultas | Escolher consultas representativas e documentar as limitações |
