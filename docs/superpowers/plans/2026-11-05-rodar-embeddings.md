# Rodar os embeddings de ponta a ponta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ter ~500 imagens do Unsplash 25k em `./imagens/` embutidas pelos 3 modelos (CLIP B/32, CLIP L/14, SigLIP), com tempos medidos e os 6 índices por modelo.

**Architecture:** Três scripts avulsos em `scripts/` (amostrar, medir, gerar) sobre o código existente (`app/embeddings.py`, `app/repo.py`, `app/carga.py`), sem alterar o fluxo de `preparar_modelo`. Cada script tem uma função pura testável e um `main()` fino. A etapa de medição roda sem banco, num subconjunto, antes da geração completa.

**Tech Stack:** Python 3.12, transformers/torch (CPU), psycopg + pgvector, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-rodar-embeddings-design.md` (complementa `2026-11-05-embeddings-e-interface-design.md`).

## Global Constraints

- Sem GPU: `DEVICE = "cpu"`; 12 núcleos, 7 GB de RAM.
- Postgres 16 + pgvector em Docker (já rodando); dimensões 512 / 768 / 768.
- Alvo: ~500 imagens em `imagens/<categoria>/`; tags vêm do nome da subpasta (`carga.sincronizar_pasta`).
- `preparar_modelo` não é alterado, salvo dependência faltando (ex.: `sentencepiece`).
- Ordem da geração completa: CLIP B/32, SigLIP, CLIP L/14.
- Testes que usam L/14 ou SigLIP levam `@pytest.mark.slow`; testes de scripts usam `embed_falso`/vetores falsos, não modelos reais.
- Código e mensagens em português, estilo do projeto.

## Review Focus

- Pasta de origem com arquivos que não são imagem (CSV, `.DS_Store`) ou imagens corrompidas: a amostragem ignora os primeiros e a medição conta os segundos como falha, sem abortar.
- Origem com menos imagens que o `N` pedido: copia todas e avisa, sem erro.
- Reexecutar a amostragem: não duplica nem sobrescreve arquivos já copiados em `imagens/`.
- Dois arquivos com o mesmo nome em subpastas diferentes da origem: não se sobrescrevem no destino.
- Falha de um modelo na geração completa (ex.: SigLIP sem `sentencepiece`): o script relata, segue para o próximo modelo e termina com código de saída diferente de zero.

---

### Task 1: Script de amostragem das imagens

**Files:**
- Create: `scripts/amostrar_imagens.py`, `scripts/__init__.py` (vazio, para os testes importarem `scripts.*`)
- Test: `tests/test_amostrar.py`
- Modify: `.gitignore` (ignorar `imagens/**/*.jpg`, `*.jpeg`, `*.png`, mantendo os `.gitkeep`), `README.md` (seção "Imagens": como baixar o Unsplash 25k do Kaggle e rodar o script)

**Interfaces:**
- Produces: `amostrar(origem: Path, destino: Path, n: int, categoria: str = "unsplash", semente: int = 42) -> int` (devolve quantas imagens copiou; ordem determinística pela semente). `main()` com argumentos `origem`, `--n` (padrão 500), `--categoria`, `--semente`, `--destino` (padrão `imagens`).

- [ ] **Step 1: Write the failing tests** em `tests/test_amostrar.py`, com `tmp_path` e imagens PIL sintéticas pequenas:
  - `test_copia_n_imagens_na_categoria`: origem com 10 jpg e 1 `.csv`; `amostrar(origem, destino, 4)` devolve 4 e `destino/unsplash/` tem 4 arquivos, nenhum `.csv`.
  - `test_menos_imagens_que_n_copia_todas`: origem com 3 jpg e `n=10` devolve 3.
  - `test_mesma_semente_mesma_amostra`: duas execuções em destinos distintos geram os mesmos nomes.
  - `test_reexecutar_nao_duplica`: segunda chamada com o mesmo destino devolve 0 e a pasta continua com 4 arquivos.
  - `test_nomes_repetidos_em_subpastas_nao_colidem`: `a/x.jpg` e `b/x.jpg` na origem, `n=2`, destino com 2 arquivos (prefixar o nome da subpasta quando colidir).
- [ ] **Step 2: Run** `pytest tests/test_amostrar.py -v` — Expected: FAIL (módulo inexistente).
- [ ] **Step 3: Implement** `amostrar` em `scripts/amostrar_imagens.py`, com o mesmo `sys.path.insert` dos outros scripts; extensões `.jpg/.jpeg/.png` (case-insensitive); varre a origem com `rglob`, ordena os caminhos, embaralha com `random.Random(semente)`, copia com `shutil.copy2` pulando destinos existentes; imprime o total copiado.
- [ ] **Step 4: Run** `pytest tests/test_amostrar.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add scripts/amostrar_imagens.py tests/test_amostrar.py .gitignore README.md && git commit -m "feat: script de amostragem das imagens do Unsplash"`

### Task 2: Obter o dataset e amostrar 500 (operacional)

**Files:**
- Nenhum arquivo de código; resultado em `imagens/unsplash/` (ignorado pelo git).

**Interfaces:**
- Consumes: `scripts/amostrar_imagens.py` (Task 1).
- Produces: ~500 imagens em `imagens/unsplash/`.

- [ ] **Step 1: Baixar o Unsplash 25k** do Kaggle (`ntsv648/unsplash-25k`) para fora do repositório, por exemplo `~/Downloads/unsplash-25k/`; é preciso conta no Kaggle (o usuário faz o login/download manual ou fornece o `kaggle` CLI configurado).
- [ ] **Step 2: Confirmar o conteúdo** com `find ~/Downloads/unsplash-25k -type f | sed 's/.*\.//' | sort | uniq -c`; anotar se há CSV de metadados (descrição/palavras-chave), pois isso decide trigram/full-text depois.
- [ ] **Step 3: Rodar** `python scripts/amostrar_imagens.py ~/Downloads/unsplash-25k --n 500`.
- [ ] **Step 4: Verificar** `find imagens -type f ! -name .gitkeep | wc -l` — Expected: 500 (ou o total disponível, se menor).

### Task 3: Script de medição dos modelos em subconjunto

**Files:**
- Create: `scripts/medir_modelos.py`
- Test: `tests/test_medir_modelos.py`

**Interfaces:**
- Consumes: `app.embeddings.embed_imagens(caminhos: list[str], abordagem: str) -> list[list[float] | None]`, `aquecer(abordagem: str)`, `dimensao(abordagem: str) -> int`; `app.config.ABORDAGENS_EMBEDDING`, `TAMANHO_LOTE`.
- Produces: `medir(abordagem: str, caminhos: list[str]) -> dict` com as chaves `abordagem`, `imagens`, `falhas`, `ms_por_imagem`, `dimensao_ok`, `norma_ok` (`norma_ok` só vale para vetores não-None); `main()` com `--n` (padrão 200), `--modelos` (padrão os 3) e `--pasta` (padrão `imagens`) que escolhe `n` imagens (semente fixa), imprime uma tabela e a estimativa `ms_por_imagem × total de imagens da pasta / 60000` em minutos.

- [ ] **Step 1: Write the failing tests** em `tests/test_medir_modelos.py`, com `monkeypatch` em `scripts.medir_modelos.embed_imagens` (e `aquecer`) devolvendo vetores falsos unitários:
  - `test_medir_calcula_campos`: 5 caminhos, vetores falsos de dimensão correta → `imagens == 5`, `falhas == 0`, `dimensao_ok` e `norma_ok` verdadeiros, `ms_por_imagem >= 0`.
  - `test_medir_conta_falhas_sem_abortar`: o falso devolve `None` em 2 de 5 → `falhas == 2`, `imagens == 5`.
  - `test_medir_detecta_dimensao_errada`: falso com dimensão errada → `dimensao_ok` falso.
- [ ] **Step 2: Run** `pytest tests/test_medir_modelos.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `medir` (chama `aquecer` antes de cronometrar, processa em lotes de `TAMANHO_LOTE` com `time.perf_counter`) e `main`.- [ ] **Step 4: Run** `pytest tests/test_medir_modelos.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add scripts tests/test_medir_modelos.py && git commit -m "feat: medição de ms/imagem dos modelos em subconjunto"`

### Task 4: Baixar os modelos e validar nas 200 imagens (operacional)

**Files:**
- Modify (condicional): `requirements.txt` (adicionar `sentencepiece` com versão fixada, se o SigLIP falhar ao carregar o tokenizador)
- Create: `docs/superpowers/specs/2026-11-05-medicoes-embeddings.md` (resultados e decisão da demo)

**Interfaces:**
- Consumes: `scripts/baixar_modelos.py`, `scripts/medir_modelos.py` (Task 3), imagens da Task 2.

- [ ] **Step 1: Baixar** os modelos com `python scripts/baixar_modelos.py` (vários GB; só L/14 e SigLIP faltam).
- [ ] **Step 2: Se o SigLIP falhar** por tokenizador/`sentencepiece`, adicionar a dependência ao `requirements.txt`, `pip install -r requirements.txt` e repetir o Step 1. Rodar também os testes lentos: `pytest tests/test_embeddings.py -m slow -v` — Expected: PASS.
- [ ] **Step 3: Medir** com `python scripts/medir_modelos.py --n 200` — Expected: os 3 modelos com `dimensao_ok` e `norma_ok` verdadeiros e `falhas` baixas.
- [ ] **Step 4: Registrar** em `docs/superpowers/specs/2026-11-05-medicoes-embeddings.md` a tabela ms/imagem, o tempo estimado da base completa por modelo e a decisão de quais modelos entram na demo ao vivo (limite sugerido: preparar ao vivo só o que levar menos de ~5 min).
- [ ] **Step 5: Commit** `git add requirements.txt docs/superpowers/specs/2026-11-05-medicoes-embeddings.md && git commit -m "docs: medições dos modelos e decisão da demo"`

### Task 5: Script de geração completa dos embeddings

**Files:**
- Create: `scripts/gerar_embeddings.py`
- Test: `tests/test_gerar_embeddings.py`

**Interfaces:**
- Consumes: `app.carga.sincronizar_pasta(raiz: str) -> int`, `app.repo.preparar_modelo(abordagem, progresso) -> dict`, `app.repo.obter_config(abordagem) -> dict | None`, `app.config.PASTA_IMAGENS`.
- Produces: `gerar(modelos: list[str], pasta: str = PASTA_IMAGENS, refazer: bool = False) -> dict[str, dict | str]` (resultado de `preparar_modelo` por modelo, ou a mensagem de erro; modelos já preparados são pulados a menos que `refazer`); `main()` com `--modelos` (padrão na ordem B/32, SigLIP, L/14), `--refazer`, `--pasta`, que imprime progresso por modelo e sai com código 1 se algum modelo falhou.

- [ ] **Step 1: Write the failing tests** em `tests/test_gerar_embeddings.py`, usando as fixtures `imagens_db`, `embed_falso` e `banco_limpo`:
  - `test_gera_modelo_e_registra_config`: `gerar(["CLIP B/32"])` → `obter_config("CLIP B/32")["total"] == 5`.
  - `test_pula_modelo_ja_preparado`: segunda chamada sem `refazer` não chama `preparar_modelo` (monkeypatch que conta chamadas); com `refazer=True` chama.
  - `test_falha_de_um_modelo_nao_impede_os_outros`: `preparar_modelo` falso que levanta para `"SigLIP"`; `gerar(["SigLIP", "CLIP B/32"])` devolve erro em um e resultado no outro.
  - `test_sem_imagens_no_catalogo_da_erro_claro`: `banco_limpo` sem imagens e pasta vazia → o resultado traz a mensagem de `preparar_modelo` ("Não há imagens no catálogo").
- [ ] **Step 2: Run** `pytest tests/test_gerar_embeddings.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `gerar`: chama `sincronizar_pasta(pasta)` uma vez, depois itera os modelos com `try/except Exception` por modelo, registrando a mensagem; progresso impresso a cada lote via o callback de `preparar_modelo`.
- [ ] **Step 4: Run** `pytest tests/test_gerar_embeddings.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add scripts/gerar_embeddings.py tests/test_gerar_embeddings.py && git commit -m "feat: script de geração completa dos embeddings"`

### Task 6: Gerar a base completa (operacional)

**Files:**
- Nenhum de código; resultado nas tabelas `emb_*` do banco `imagens`.

**Interfaces:**
- Consumes: `scripts/gerar_embeddings.py` (Task 5), decisão da Task 4.

- [ ] **Step 1: Confirmar o banco** com `docker compose ps` (Postgres no ar) e o schema aplicado (`scripts/init.sh --sem-modelos`, idempotente).
- [ ] **Step 2: Gerar** com `python scripts/gerar_embeddings.py` (B/32, SigLIP, L/14, nessa ordem). Rodar em um terminal que fique aberto; uma falha em um modelo não perde os já preparados.
- [ ] **Step 3: Verificar** no banco: para cada modelo, `SELECT count(*) FROM emb_<modelo>` igual ao total de imagens legíveis e 6 índices em `pg_indexes` (`SELECT count(*) FROM pg_indexes WHERE tablename = 'emb_clip_b32'` → 6); `SELECT abordagem, total, tempo_geracao_s FROM config_embedding` com os 3 modelos.
- [ ] **Step 4: Teste final de ponta a ponta** abrindo `streamlit run app/ui.py`, escolhendo cada modelo, digitando uma consulta em inglês (ex.: "dog on the beach") e conferindo 3 resultados com score e tempos.
