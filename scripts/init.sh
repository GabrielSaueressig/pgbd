#!/usr/bin/env bash
# Inicializa o ambiente local. Idempotente: pode rodar várias vezes.
# Uso: scripts/init.sh [--sem-modelos]
set -euo pipefail

cd "$(dirname "$0")/.."

BAIXAR_MODELOS=1
[[ "${1:-}" == "--sem-modelos" ]] && BAIXAR_MODELOS=0

echo "==> .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "criado a partir de .env.example"
else
  echo "já existe"
fi
set -a; source .env; set +a

echo "==> Postgres (docker compose)"
docker compose up -d

echo "==> Aguardando o Postgres"
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker compose exec -T postgres pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB"

echo "==> Schema"
docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" < schema.sql

echo "==> Python (.venv)"
if command -v uv >/dev/null 2>&1; then
  [[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
  uv pip install --quiet --python .venv/bin/python -r requirements.txt
else
  [[ -x .venv/bin/pip ]] || { rm -rf .venv; python3 -m venv .venv; }
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

if [[ $BAIXAR_MODELOS -eq 1 ]]; then
  echo "==> Modelos"
  .venv/bin/python scripts/baixar_modelos.py
else
  echo "==> Modelos: pulado (--sem-modelos)"
fi

cat <<MSG

Pronto. Para iniciar a interface:
  source .venv/bin/activate
  streamlit run app/ui.py
MSG
