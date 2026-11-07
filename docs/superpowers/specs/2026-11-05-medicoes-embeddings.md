# Medições dos embeddings (CPU, 12 núcleos, 73 imagens)

| Modelo | Dimensão | ms/imagem | Geração completa (preparar_modelo, com 6 índices) |
|---|---|---|---|
| CLIP B/32 | 512 | 173 | 11 s |
| SigLIP | 768 | 199 | 13 s |
| CLIP L/14 | 768 | 522 | 41 s |

- Falhas: 0 em todos; dimensão e norma ≈ 1 conferidas.
- Dependência adicionada: `sentencepiece==0.2.0` (o SigLIP precisa; versões mais novas exigem protobuf ≥ 7, incompatível com o Streamlit).

## Decisão para a demo ao vivo

Os três modelos preparam em menos de 1 minuto, bem abaixo do limite de ~5 min, então todos podem ser preparados ao vivo.

## Limitação

Com 73 imagens, exata, HNSW e IVFFlat tendem a ter tempos e resultados quase idênticos (recall ~100%); ver aviso no README.
