# Design: geração de embeddings e interface de medição

Complementa `especificacao-busca-imagens.md` (v4) e `plano-implementacao.md`. Cobre os passos 2, 3 e 5 do plano, mais um stub de busca.

## Intenção
Permitir gerar os embeddings das imagens de `./imagens` (Unsplash 25k, ~5k imagens) com 3 modelos e testar tudo pela interface Streamlit, medindo tempo de geração e de busca. Trabalho acadêmico: o foco é comparar abordagens, não um sistema de uso real.

## Objetivos
- `app/embeddings.py`: carregar CLIP B/32, CLIP L/14 e SigLIP e gerar vetores de imagem e de texto normalizados (L2).
- `app/repo.py`: `preparar_modelo(abordagem, progresso)` cria `emb_*`, insere em lotes, mede `tempo_geracao_s`, cria os 6 índices e grava `config_embedding`. Também `modelo_preparado` e `obter_config(abordagem)`.
- `app/ui.py` conforme o plano v4, passo 5: linha de controles, k=3, botão "Preparar modelo", resultado com score, tempos e `EXPLAIN`.
- `app/busca.py` mínimo, com o contrato do plano (`buscar`, `explicar`), para a UI funcionar de ponta a ponta. A Pessoa C o substitui depois.

## Fora do escopo
- Benchmark (`benchmark.py`), trigram e full-text, upload de imagens, GPU.
- Implementação definitiva da busca (fica com a Pessoa C).

## Anti-objetivos
- Tempos de modelos diferentes que não sejam comparáveis entre si.
- Perder horas de geração por uma falha no meio.

## Restrições
- Postgres 16 + pgvector em Docker; a dimensão é fixa por tabela (512 / 768 / 768).
- Sem GPU: inferência na CPU.
- Origem das imagens: arquivos em `./imagens`, caminho no Postgres, tags vindas da subpasta (decisão confirmada).

## Abordagens consideradas
- **A. Só `transformers` (escolhida):** `AutoModel` + `AutoProcessor` para os 3 modelos, com `get_image_features` / `get_text_features` e normalização própria. Um caminho de código e tempos comparáveis.
- **B. Como na spec** (`sentence-transformers` para CLIP, `transformers` para SigLIP): dois caminhos de código e pré-processamento diferente, o que torna os tempos menos comparáveis. Descartada.

## Decisões
- Inferência em lotes de `TAMANHO_LOTE` (16) com `torch.no_grad()`. Commit por lote, para a barra de progresso ser real e a falha não perder o já gerado.
- Os 6 índices (HNSW e IVFFlat × cosseno, L2, produto interno) são criados só no final; o IVFFlat só com a tabela populada, `lists ≈ √N`.
- `tempo_geracao_s` mede inferência + inserção da base inteira.
- `aquecer(abordagem)` roda uma inferência descartável ao selecionar o modelo, fora das medições.
- Stub de `busca.py`: SQL básico, estratégia por `SET LOCAL`, mediana de 5 execuções com a 1ª à parte.
- Decisão minha: se a preparação falhar no meio, a tabela `emb_*` parcial é descartada, e a próxima tentativa recomeça do zero.

## Deixado para quem implementa
- Detalhes de dtype, tamanho do `lists` em subconjuntos e ajuste de `maintenance_work_mem`.
- Testes: dimensão esperada, norma ≈ 1, N linhas, 6 índices presentes, tempo gravado.
