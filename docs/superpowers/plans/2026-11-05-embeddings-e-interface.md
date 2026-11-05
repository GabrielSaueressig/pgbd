# Embeddings e interface de medição: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar embeddings das imagens de `./imagens` com 3 modelos, guardá-los em tabelas `emb_*` indexadas no Postgres/pgvector e testar tudo pela UI Streamlit.

**Architecture:** `embeddings.py` carrega os 3 modelos pela mesma API `transformers` (abordagem A do design) e devolve vetores L2-normalizados. `repo.py` prepara cada modelo (tabela, lotes com commit, tempo, 6 índices, `config_embedding`). `busca.py` é um stub funcional com o contrato do plano, e `ui.py` o consome.

**Tech Stack:** Python 3.12, transformers 4.46.3, torch 2.5.1 (CPU), psycopg 3, pgvector-python, Streamlit 1.39, pytest (novo, dev).

**Spec:** `docs/superpowers/specs/2026-11-05-embeddings-e-interface-design.md` (e `especificacao-busca-imagens.md` v4)

## Global Constraints

- Banco: `pgvector/pgvector:pg16`, conexão via `DATABASE_URL` (`app/db.py:conexao()`, autocommit).
- Tabelas: `emb_clip_b32` (512), `emb_clip_l14` (768), `emb_siglip` (768); colunas `imagem_id INT PRIMARY KEY REFERENCES imagem(id) ON DELETE CASCADE`, `vetor vector(dim) NOT NULL`.
- 6 índices por tabela: HNSW e IVFFlat × `vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`. IVFFlat só após a carga, `lists ≈ √N`.
- `TAMANHO_LOTE = 16`; commit por lote; sem GPU (`device="cpu"`).
- `config_embedding(abordagem PK, tabela, dimensao, total, tempo_geracao_s, gerado_em)`; uma linha por modelo.
- UI sem sidebar, k fixo em 3, métricas Cosseno `<=>`, L2 `<->`, Produto interno `<#>`.
- Busca: `SET LOCAL enable_*` força a estratégia; 5 execuções, mediana e 1ª à parte; embedding da consulta fora da medição.
- Em falha da preparação, a tabela `emb_*` parcial é descartada.

## Review Focus

- Pasta `imagens/` vazia: a UI avisa que não há imagens e o "Preparar modelo" não roda com N=0 (IVFFlat exige linhas).
- Imagem corrompida ou em modo CMYK/RGBA/P: é convertida para RGB; se não abrir, é pulada e contada, sem derrubar o lote.
- Texto de consulta vazio ou só espaços: a busca não executa.
- Preparar de novo um modelo já preparado: recria a tabela do zero, sem duplicar nem deixar índices antigos.
- Caminho de imagem que sumiu do disco depois da carga: a UI mostra um aviso, não uma exceção.

---

### Task 1: Config, dependências e infra de testes

**Files:**
- Modify: `app/config.py`, `scripts/baixar_modelos.py`, `requirements.txt`
- Create: `tests/__init__.py`, `tests/conftest.py`, `pytest.ini`

**Interfaces:**
- Produces: `ABORDAGENS_EMBEDDING: dict[str, {"modelo": str, "dimensao": int, "tabela": str}]` com chaves `"CLIP B/32"` (`openai/clip-vit-base-patch32`, 512, `emb_clip_b32`), `"CLIP L/14"` (`openai/clip-vit-large-patch14`, 768, `emb_clip_l14`), `"SigLIP"` (`google/siglip-base-patch16-224`, 768, `emb_siglip`); `ABORDAGENS_BUSCA = ["Vetorial exata", "Vetorial HNSW", "Vetorial IVFFlat"]`; `OPCLASS_POR_METRICA: dict[str, str]` (`"Cosseno"→"vector_cosine_ops"`, etc.); `K_FIXO = 3`; fixture `conn` (conexão ao banco, `pytest.skip` se `DATABASE_URL` indisponível).

- [ ] **Step 1:** Em `requirements.txt`, remover `sentence-transformers` e adicionar `pytest==8.3.3`. Em `scripts/baixar_modelos.py`, trocar o download para `AutoProcessor`/`AutoModel` para os 3 modelos.
- [ ] **Step 2:** Editar `app/config.py` conforme Interfaces, mantendo `METRICAS`, `TAMANHO_LOTE` e `PASTA_IMAGENS`.
- [ ] **Step 3:** Criar `pytest.ini` (`testpaths = tests`) e `tests/conftest.py` com a fixture `conn` (usa `app.db.conexao`) e a fixture `imagens_tmp` (cria 5 JPEGs sintéticos de cores distintas em `tmp_path` e devolve a lista de caminhos).
- [ ] **Step 4:** Run `.venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q`. Expected: `no tests ran`, sem erro de importação.
- [ ] **Step 5:** Commit `chore: config dos 3 modelos via transformers e pytest`.

### Task 2: `embeddings.py`

**Files:**
- Modify: `app/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `ABORDAGENS_EMBEDDING` (Task 1).
- Produces: `carregar_modelo(abordagem: str) -> tuple[model, processor]` (cache `lru_cache(maxsize=3)`); `embed_imagens(caminhos: list[str], abordagem: str) -> list[list[float] | None]` (lote; `None` para imagem ilegível); `embed_imagem(caminho: str, abordagem: str) -> list[float]`; `embed_texto(texto: str, abordagem: str) -> list[float]`; `aquecer(abordagem: str) -> None`; `dimensao(abordagem: str) -> int`.

- [ ] **Step 1: Failing tests** (`@pytest.mark.parametrize("abordagem", ["CLIP B/32"])` por padrão; marcar os outros 2 como `slow`, rodados com `-m slow`):
  - `test_dimensao_e_norma_imagem`: `len(embed_imagem(p, a)) == dimensao(a)` e `abs(norma - 1.0) < 1e-4`.
  - `test_dimensao_e_norma_texto`: o mesmo para `embed_texto("praia", a)`.
  - `test_lote_devolve_none_para_imagem_corrompida`: lote `[valida, arquivo_de_texto.jpg]` devolve `[vetor, None]`.
  - `test_modo_cmyk_rgba_convertido`: imagem em `CMYK` e em `RGBA` geram vetor válido.
  - `test_texto_relaciona_imagem`: com imagem vermelha e imagem azul, `embed_texto("a red image")` tem maior cosseno com a vermelha.
- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_embeddings.py -v`. Expected: FAIL (`NotImplementedError`).
- [ ] **Step 3:** Implementar. CLIP usa `CLIPModel`/`CLIPProcessor` e SigLIP usa `AutoModel`/`AutoProcessor`, ambos com `get_image_features` / `get_text_features`. `torch.no_grad()`, `model.eval()`, CPU, normalização L2 e `Image.open(...).convert("RGB")` com `try/except`. Texto do SigLIP com `padding="max_length"`, `max_length=64`.
- [ ] **Step 4:** Run o mesmo comando. Expected: PASS (o 1º run baixa o CLIP B/32).
- [ ] **Step 5:** Commit `feat: geração de embeddings de imagem e texto`.

### Task 3: `repo.py`: preparar modelo

**Files:**
- Modify: `app/repo.py` (remover `aplicar_abordagem` e a tabela única `embedding`), `app/db.py` (adicionar `criar_schema()`)
- Test: `tests/test_repo.py`

**Interfaces:**
- Consumes: `embed_imagens`, `dimensao` (Task 2); `ABORDAGENS_EMBEDDING`, `OPCLASS_POR_METRICA`, `TAMANHO_LOTE` (Task 1).
- Produces: `criar_schema() -> None` (executa `schema.sql`); `preparar_modelo(abordagem: str, progresso: Callable[[int, int], None] | None = None) -> dict` (devolve `{"total": int, "puladas": int, "tempo_geracao_s": float}`); `modelo_preparado(abordagem: str) -> bool`; `obter_config(abordagem: str) -> dict | None` (colunas de `config_embedding`); `total_imagens() -> int`.

- [ ] **Step 1: Failing tests.** Usam `conn` + 5 imagens sintéticas inseridas em `imagem`, e o `monkeypatch` de `embed_imagens` por um stub rápido que devolve vetores unitários de `dimensao` (dispensa o modelo). Um teste de integração real usa `CLIP B/32`.
  - `test_cria_tabela_com_n_linhas_e_6_indices`: `SELECT count(*)` = 5 e 6 linhas em `pg_indexes` para a tabela, entre elas `hnsw` e `ivfflat`.
  - `test_grava_config_e_tempo`: `obter_config(a)["tempo_geracao_s"] > 0` e `total == 5`.
  - `test_progresso_chamado_por_lote`: com `TAMANHO_LOTE=2`, o callback recebe `(2,5), (4,5), (5,5)`.
  - `test_repreparar_nao_duplica`: preparar duas vezes → 5 linhas e 6 índices.
  - `test_imagem_ilegivel_e_pulada`: o stub devolve `None` em uma imagem → `puladas == 1`, 4 linhas.
  - `test_sem_imagens_levanta_erro_claro`: `ValueError` quando `imagem` está vazia.
  - `test_falha_no_meio_descarta_tabela`: o stub levanta `RuntimeError` no 2º lote → a tabela não existe e `modelo_preparado` é `False`.
- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_repo.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implementar `preparar_modelo`: `DROP` + `CREATE` da tabela; para cada lote, `embed_imagens` e INSERT com commit (cada `conn.execute` já é autocommit); medir com `time.perf_counter` desde o 1º lote até o fim da inserção; depois criar os 6 índices (`IVFFlat` com `lists = max(1, round(sqrt(N)))`; `SET maintenance_work_mem = '256MB'` na sessão) e fazer UPSERT em `config_embedding`. `except`: `DROP TABLE IF EXISTS` e relevantar. Chamar `criar_schema()` na inicialização da UI.
- [ ] **Step 4:** Run o mesmo comando. Expected: PASS.
- [ ] **Step 5:** Commit `feat: preparar modelo com lotes, tempo e 6 índices`.

### Task 4: `busca.py` (stub com o contrato)

**Files:**
- Modify: `app/busca.py`
- Test: `tests/test_busca.py`

**Interfaces:**
- Consumes: `embed_texto`, `obter_config`/`modelo_preparado`, `METRICAS`, `OPCLASS_POR_METRICA`.
- Produces: `buscar(texto: str, modelo_busca: str, k: int, metrica: str, abordagem: str) -> dict` com `{"resultados": [{"caminho","nome","score"}], "tempo_ms": float (mediana de 5), "tempo_ms_1a": float, "plano": str}`; `explicar(texto, modelo_busca, k, metrica, abordagem) -> str` (`EXPLAIN (ANALYZE)`). `ValueError` para texto vazio ou modelo não preparado.

- [ ] **Step 1: Failing tests** (reusam a preparação da Task 3 com o stub de vetores):
  - `test_exata_usa_seq_scan`: o `plano` de "Vetorial exata" contém `Seq Scan`.
  - `test_hnsw_e_ivfflat_usam_indice`: o plano contém `hnsw` / `ivfflat` (no stub, se o planner insistir no Seq Scan com N=5, o teste insere N≥200 vetores aleatórios).
  - `test_score_decrescente_e_k_resultados`: `len == k` e scores ordenados por `ORDER BY` da métrica (cosseno: `1 - dist`; L2: `-dist`; produto interno: `-(dist)`; sempre decrescente).
  - `test_texto_vazio_levanta`: `buscar("   ", ...)` → `ValueError`.
  - `test_modelo_nao_preparado_levanta`.
- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_busca.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implementar. Estratégia por `SET LOCAL` em uma transação: exata → `enable_indexscan=off`, `enable_bitmapscan=off`; HNSW/IVFFlat → `enable_seqscan=off`; `ef_search`/`probes` com os padrões de `config.py`. O nome da tabela vem de `config.py` (não de entrada do usuário). O embedding da consulta é calculado antes do cronômetro.
- [ ] **Step 4:** Run o mesmo comando. Expected: PASS.
- [ ] **Step 5:** Commit `feat: busca vetorial com estratégia forçada (contrato do plano)`.

### Task 5: `ui.py`

**Files:**
- Modify: `app/ui.py`
- Test: `tests/test_ui.py` (`streamlit.testing.v1.AppTest`)

**Interfaces:**
- Consumes: `preparar_modelo`, `modelo_preparado`, `obter_config`, `total_imagens`, `aquecer`, `buscar`, `sincronizar_pasta`, `criar_schema`.

- [ ] **Step 1: Failing tests** com `AppTest.from_file("app/ui.py")` e as funções do banco/modelo trocadas por stubs:
  - `test_sem_imagens_mostra_aviso`: `total_imagens() == 0` → existe `st.warning` e não há botão de busca ativo.
  - `test_modelo_nao_preparado_mostra_botao_preparar`: aparece o botão "Preparar modelo" e a busca fica desabilitada.
  - `test_busca_mostra_3_imagens_e_tempos`: com `buscar` stubado, aparecem 3 imagens, "Tempo de busca", "Tempo de gerar embeddings" e um expander "EXPLAIN".
  - `test_arquivo_ausente_nao_quebra`: caminho inexistente → aviso e a página não levanta exceção.
- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_ui.py -v`. Expected: FAIL.
- [ ] **Step 3:** Reescrever `ui.py` sem sidebar. Linha de controles: selects de embedding, busca e métrica, e o campo de texto. Resultado: 3 imagens com `st.image` e score, faixa com mediana e 1ª execução, tempo de geração (`config_embedding.tempo_geracao_s`), índice usado, e `st.expander("EXPLAIN")`. "Preparar modelo" com `st.progress`. `aquecer(abordagem)` com `st.cache_resource` ao selecionar o modelo.
- [ ] **Step 4:** Run `.venv/bin/pytest -q`. Expected: tudo PASS.
- [ ] **Step 5:** Verificação manual: `scripts/init.sh --sem-modelos` (se o banco não estiver no ar) e `.venv/bin/streamlit run app/ui.py` com ~20 imagens em `imagens/`; preparar CLIP B/32, buscar "dog" nos 3 modos × 3 métricas e conferir o `EXPLAIN`. Registrar o `ms/imagem` observado em `README.md`.
- [ ] **Step 6:** Commit `feat: interface Streamlit de medição`.

---

**Cobertura do design:** embeddings (Task 2), preparar modelo (Task 3), stub de busca (Task 4), UI (Task 5), config e modelos via `transformers` (Task 1).
