# Plano de implementação: interface de medição (spec v4)

Ordem pensada para que cada passo seja testável sozinho. Contratos da seção 12 da especificação continuam valendo, com as alterações indicadas.

## 0. Pré-requisitos
- [ ] Baixar o Unsplash 25k, separar ~5k imagens em `./imagens` e verificar se há metadados (descrição/palavras-chave).
- [ ] `docker compose up -d`, `python -m venv`, `pip install -r requirements.txt`.
- [ ] Baixar os 3 modelos antes (cache local) e medir `ms/imagem` na CPU para confirmar a estimativa de tempo.

- [ ] `scripts/init.sh` idempotente (Postgres 16 + pgvector): `.env` a partir do exemplo, `docker compose up -d`, espera de readiness, venv + `requirements.txt`, extensões e `schema.sql`, pré-download dos modelos. Todo integrante do grupo roda só isso para ficar com o mesmo ambiente. Documentar no README.

## 1. Schema e config (`app/db.py`, `app/config.py`, `schema.sql`)
- [ ] `schema.sql` com `imagem` e `config_embedding` (uma linha por modelo: `abordagem` PK, `tabela`, `dimensao`, `total`, `tempo_geracao_s`, `gerado_em`). Hoje nenhum schema existe.
- [ ] `db.criar_schema()` executa o SQL; chamado na inicialização da UI.
- [ ] `config.py`: `ABORDAGENS_BUSCA` reduzido a 3; adicionar `TABELA_POR_ABORDAGEM`, `OPCLASS_POR_METRICA` e parâmetros padrão (`ef_search`, `probes`, `lists`).

## 2. Embeddings (`app/embeddings.py`)
- [ ] `carregar_modelo` (cache) para CLIP via sentence-transformers e SigLIP via transformers.
- [ ] `embed_imagem`, `embed_texto` normalizados; `aquecer(abordagem)` roda uma inferência descartável.
- [ ] Teste: vetor com a dimensão esperada e norma ≈ 1.

## 3. Preparar modelo (`app/repo.py`)
- [ ] `preparar_modelo(abordagem, progresso)`: cria `emb_*`, insere em lotes, mede `tempo_geracao_s`, cria os 6 índices (IVFFlat depois do `INSERT`, `lists ≈ √N`), grava `config_embedding`.
- [ ] `modelo_preparado(abordagem)` e `obter_config(abordagem)`.
- [ ] Teste: tabela com N linhas, 6 índices presentes, tempo gravado.

## 4. Busca (`app/busca.py`)
- [ ] `buscar(texto, modelo_busca, k, metrica, abordagem)` devolve `{caminho, nome, score, tempo_ms, tempo_ms_1a}` e o plano.
- [ ] Estratégia por `SET LOCAL`; operador conforme a métrica (`<=>`, `<->`, `<#>`); score coerente com a métrica, em ordem decrescente.
- [ ] Mediana de 5 execuções, 1ª à parte; embedding da consulta fora da medição.
- [ ] `explicar(...)` com `EXPLAIN (ANALYZE)`; teste que a exata usa `Seq Scan` e HNSW/IVFFlat usam o índice esperado.

## 5. Interface (`app/ui.py`)
- [ ] Linha de controles (embedding, busca, métrica, texto); k = 3.
- [ ] Modelo sem tabela: botão "Preparar modelo" com `st.progress`.
- [ ] Resultado: 3 imagens com score; faixa com tempo de busca, tempo de geração da base e índice usado; expander com `EXPLAIN`.
- [ ] Aquecimento do modelo ao selecioná-lo.

## 6. Benchmark (`benchmark.py`)
- [ ] Conjunto de consultas de texto (20 a 30) sem gabarito manual.
- [ ] Para cada modelo × métrica × índice: latência média/p95, **recall@3 e @10 contra a exata**, `pg_relation_size`.
- [ ] Varredura de `ef_search`, `probes` e `lists`; tabela comparativa (CSV/Markdown).
- [ ] Trigram e full-text apenas se houver metadados.

## 7. Verificação e demo
- [ ] Rodar uma busca por combinação e conferir o `EXPLAIN`.
- [ ] Capturas de tela e vídeo de reserva (plano B da seção 15).

## Divisão sugerida
| Passos | Responsável |
|---|---|
| 1, 2, 3 | Pessoa B |
| 4, 5 | Pessoa C |
| 6, 7 | Pessoa A |

## Riscos novos
- 18 índices HNSW/IVFFlat em ~5k vetores: construção rápida, mas confirmar o uso de memória (7 GB de RAM).
- Com ~5k imagens, a diferença de tempo entre exata e ANN pode ser pequena; o benchmark deve escalar subconjuntos (1k, 2,5k, 5k) para mostrar a curva.
- L/14 e SigLIP na CPU: medir antes de decidir se entram na demo ao vivo.
