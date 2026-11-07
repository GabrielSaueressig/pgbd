#!/usr/bin/env bash
# Setup completo e execução em um comando. Idempotente.
# Uso: scripts/rodar.sh [--sem-modelos] [--sem-ui]
set -euo pipefail

cd "$(dirname "$0")/.."

ARGS_INIT=()
ABRIR_UI=1
for a in "$@"; do
  case "$a" in
    --sem-modelos) ARGS_INIT+=(--sem-modelos) ;;
    --sem-ui) ABRIR_UI=0 ;;
    *) echo "argumento desconhecido: $a" >&2; exit 2 ;;
  esac
done

echo "==> Imagens"
if [[ -z "$(find imagens -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | head -1)" ]]; then
  [[ -f dados/imagens-unsplash.zip ]] || { echo "dados/imagens-unsplash.zip não encontrado e imagens/ está vazia." >&2; exit 1; }
  mkdir -p imagens
  unzip -q -n dados/imagens-unsplash.zip -d imagens
  echo "extraídas de dados/imagens-unsplash.zip"
else
  echo "já existem"
fi

scripts/init.sh "${ARGS_INIT[@]}"

echo "==> Embeddings (pula os modelos já preparados)"
.venv/bin/python scripts/gerar_embeddings.py

if [[ $ABRIR_UI -eq 1 ]]; then
  echo "==> Interface"
  exec .venv/bin/streamlit run app/ui.py
fi
