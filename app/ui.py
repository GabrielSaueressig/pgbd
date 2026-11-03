"""Interface Streamlit: página única com select de embedding e select de busca."""

import streamlit as st

from app.carga import sincronizar_pasta
from app.config import ABORDAGENS_BUSCA, ABORDAGENS_EMBEDDING, METRICAS
from app.repo import aplicar_abordagem, obter_config

st.set_page_config(page_title="Busca de imagens com pgvector", layout="wide")

if "carga_feita" not in st.session_state:
    sincronizar_pasta("imagens")
    st.session_state["carga_feita"] = True

with st.sidebar:
    st.header("Embeddings")
    abordagem = st.selectbox("Abordagem de embedding", list(ABORDAGENS_EMBEDDING))
    if st.button("Gerar embeddings"):
        barra = st.progress(0.0, text="Gerando embeddings...")
        aplicar_abordagem(
            abordagem,
            progresso=lambda feitas, total: barra.progress(feitas / total, text=f"{feitas}/{total}"),
        )
        st.success("Embeddings gerados")
    config = obter_config()
    st.caption(f"Ativa: {config['modelo'] or 'nenhuma'} | dimensão: {config['dimensao']} | total: {config['total']}")

st.title("Busca de imagens por texto")

col_texto, col_k = st.columns([4, 1])
texto = col_texto.text_input("Descreva a imagem")
k = col_k.number_input("Resultados (k)", min_value=1, max_value=50, value=5)

col_busca, col_metrica = st.columns(2)
modelo_busca = col_busca.selectbox("Abordagem de busca", ABORDAGENS_BUSCA)
metrica = col_metrica.selectbox("Métrica (vetorial)", list(METRICAS))

if st.button("Buscar") and texto:
    # Área 3: chamar busca.buscar(texto, modelo_busca, k, metrica=metrica) e exibir a grade
    st.info("Busca ainda não implementada (Área 3).")
