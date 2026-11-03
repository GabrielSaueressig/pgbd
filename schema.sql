-- Schema inicial (idempotente). As tabelas emb_* são criadas por app/repo.py ao preparar cada modelo.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- array_to_string é STABLE; o wrapper IMMUTABLE permite usá-lo em coluna gerada e em índice
CREATE OR REPLACE FUNCTION imagem_texto(nome TEXT, tags TEXT[]) RETURNS TEXT
  LANGUAGE sql IMMUTABLE PARALLEL SAFE
  AS $$ SELECT coalesce(nome, '') || ' ' || coalesce(array_to_string(tags, ' '), '') $$;

CREATE TABLE IF NOT EXISTS imagem (
  id       SERIAL PRIMARY KEY,
  nome     TEXT NOT NULL,
  caminho  TEXT NOT NULL UNIQUE,
  tags     TEXT[] DEFAULT '{}',
  tsv      tsvector GENERATED ALWAYS AS (to_tsvector('simple', imagem_texto(nome, tags))) STORED
);
CREATE INDEX IF NOT EXISTS imagem_tsv_gin ON imagem USING gin (tsv);
CREATE INDEX IF NOT EXISTS imagem_trgm_gin
  ON imagem USING gin (imagem_texto(nome, tags) gin_trgm_ops);

-- Uma linha por modelo preparado
CREATE TABLE IF NOT EXISTS config_embedding (
  abordagem        TEXT PRIMARY KEY,
  tabela           TEXT NOT NULL,
  dimensao         INT NOT NULL,
  total            INT DEFAULT 0,
  tempo_geracao_s  DOUBLE PRECISION,
  gerado_em        TIMESTAMPTZ
);
