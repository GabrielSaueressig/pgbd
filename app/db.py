"""Conexão com o PostgreSQL e criação das extensões."""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def conexao() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(conn)
    return conn


def criar_extensoes(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
