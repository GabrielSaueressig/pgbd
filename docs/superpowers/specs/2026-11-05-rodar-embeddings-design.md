# Design: rodar os embeddings de ponta a ponta

Complementa `2026-11-05-embeddings-e-interface-design.md`. O código de `app/embeddings.py` e `app/repo.py` já existe; este design cobre obter os dados, validar os modelos, medir e gerar a base.

## Intenção
Ter as ~500 imagens do Unsplash 25k embutidas pelos 3 modelos (CLIP B/32, CLIP L/14, SigLIP) em tabelas `emb_*` com os 6 índices, para a interface e o benchmark usarem dados reais. Trabalho acadêmico: o foco é comparar abordagens.

## Objetivos
- `./imagens/<categoria>/` com ~500 imagens amostradas do Unsplash 25k.
- Os 3 modelos baixados e carregando na CPU.
- ms/imagem medido por modelo; decisão registrada sobre quais modelos entram na demo ao vivo.
- Os 3 modelos preparados (`preparar_modelo`) na base completa.

## Não objetivos
- Mudar o código de embeddings ou o fluxo de `preparar_modelo`, salvo o que a validação revelar (ex.: dependência faltando).
- Benchmark, GPU, upload de imagens.

## Anti-objetivos
- Perder horas de geração por falha tardia: por isso a validação em subconjunto vem antes da geração completa.

## Restrições
- Sem GPU, 12 núcleos, 7 GB de RAM; Postgres 16 + pgvector em Docker (já rodando).
- O dataset vem do Kaggle (conta necessária); ainda não foi baixado.

## Abordagens consideradas
- **A. Em etapas, medindo antes (escolhida):** baixar e amostrar; validar os 3 modelos em ~200 imagens; gerar a base completa um modelo por vez.
- **B. Só B/32 na base completa:** rápida, mas problemas do SigLIP aparecem tarde. Descartada.
- **C. Tudo em lote sem acompanhar:** `preparar_modelo` descarta a tabela se falhar, então uma falha tardia no L/14 perde tudo. Descartada.

## Passos
1. Baixar o Unsplash 25k (Kaggle) e criar um script de amostragem que copia ~500 imagens para `imagens/<categoria>/` (as subpastas viram tags, como já faz `carga.sincronizar_pasta`).
2. Rodar `scripts/baixar_modelos.py` para L/14 e SigLIP (B/32 já está no cache).
3. Validar os 3 modelos em ~200 imagens: carregam, dimensões 512/768/768, norma ≈ 1, e medir ms/imagem. Se o SigLIP falhar por tokenizador, adicionar `sentencepiece` ao `requirements.txt`.
4. Com as medições, estimar o tempo da base completa e decidir quais modelos entram na demo ao vivo.
5. Rodar `sincronizar_pasta` e `preparar_modelo` na base completa, um modelo por vez (B/32, SigLIP, L/14), conferindo 6 índices e `tempo_geracao_s` gravado.

## Decisões minhas (a confirmar)
- Consultas da demo em inglês, porque os modelos CLIP são em inglês; tradução de consultas fica fora do escopo.
- Os ~200 imagens de validação saem da própria amostra de 5k, em um banco limpo antes da geração completa.
- Número de categorias e critério de amostragem (aleatório simples) ficam a cargo de quem implementa.

## Deixado para quem implementa
- Formato exato do script de amostragem e de medição (podem ser scripts avulsos em `scripts/`).
- Ajuste de `maintenance_work_mem` se a criação dos índices estourar a memória.
