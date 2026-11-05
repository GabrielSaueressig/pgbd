"""Interface Streamlit: escolher modelo e busca, preparar embeddings e medir tempos."""

import sys
from pathlib import Path

# `streamlit run app/ui.py` põe app/ no sys.path, não a raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import busca, carga, db, embeddings, repo
from app.config import ABORDAGENS_BUSCA, ABORDAGENS_EMBEDDING, K_FIXO, METRICAS, PASTA_IMAGENS

st.set_page_config(page_title="Busca de imagens com pgvector", layout="wide")


@st.cache_resource(show_spinner="Preparando o ambiente...")
def inicializar() -> int:
    db.criar_schema()
    return carga.sincronizar_pasta(PASTA_IMAGENS)


@st.cache_resource(show_spinner="Carregando o modelo...")
def aquecer(abordagem: str) -> bool:
    embeddings.aquecer(abordagem)
    return True


inicializar()
st.title("Busca de imagens por texto")

total = repo.total_imagens()
if total == 0:
    st.warning(f"Não há imagens no catálogo. Coloque arquivos em ./{PASTA_IMAGENS}/<categoria>/ e recarregue.")

c1, c2, c3 = st.columns(3)
abordagem = c1.selectbox("Abordagem de embedding", list(ABORDAGENS_EMBEDDING))
modo = c2.selectbox("Abordagem de busca", ABORDAGENS_BUSCA)
metrica = c3.selectbox("Métrica", list(METRICAS))

preparado = total > 0 and repo.modelo_preparado(abordagem)
if total > 0 and not preparado:
    st.info(f"O modelo {abordagem} ainda não gerou embeddings para as {total} imagens.")
    if st.button("Preparar modelo", key="preparar"):
        barra = st.progress(0.0, text="Gerando embeddings...")
        try:
            r = repo.preparar_modelo(
                abordagem,
                progresso=lambda f, t: barra.progress(f / t, text=f"{f}/{t} imagens"),
            )
        except Exception as e:
            barra.empty()
            st.error(f"Falha ao preparar o modelo: {e}")
        else:
            st.success(f"Pronto: {r['total']} embeddings em {r['tempo_geracao_s']:.1f} s"
                       + (f" ({r['puladas']} imagens ilegíveis puladas)" if r["puladas"] else ""))
            st.rerun()

if preparado:
    aquecer(abordagem)

texto = st.text_input("Descreva a imagem", disabled=not preparado)
clicou = st.button("Buscar", key="buscar", disabled=not preparado)

if clicou:
    if not texto.strip():
        st.warning("Digite um texto para buscar.")
    else:
        try:
            r = busca.buscar(texto, modo, K_FIXO, metrica, abordagem)
        except ValueError as e:
            st.error(str(e))
        else:
            cfg = repo.obter_config(abordagem)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tempo de busca", f"{r['tempo_ms']:.2f} ms", help="Mediana de 5 execuções")
            m2.metric("1ª execução", f"{r['tempo_ms_1a']:.2f} ms")
            m3.metric("Tempo de gerar embeddings", f"{cfg['tempo_geracao_s']:.1f} s", help="Base inteira, última geração")
            m4.metric("Índice usado", r["indice"] or "nenhum (varredura)")
            colunas = st.columns(K_FIXO)
            for col, item in zip(colunas, r["resultados"]):
                with col:
                    if Path(item["caminho"]).is_file():
                        st.image(item["caminho"], caption=f"{item['nome']} · score {item['score']:.4f}")
                    else:
                        st.warning(f"Arquivo não encontrado: {item['caminho']}")
            with st.expander("EXPLAIN"):
                st.code(r["plano"])

st.divider()
st.subheader("Relatório")
if st.button("Testar todas as combinações e gerar relatório", key="relatorio"):
    # Futuro: percorrer modelo × busca × métrica (ligar ao benchmark.py), acumular os
    # resultados e gerar o relatório acumulado.
    st.info("Ainda não implementado: será feito futuramente.")
