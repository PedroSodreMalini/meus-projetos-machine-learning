import streamlit as st
import pandas as pd

# Carregar Dados e colocar no Cache do Streamlit
@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset/laptops-with-cluster.csv")

df = carregar_dados()

st.sidebar.header("Filtros")

model = st.sidebar.selectbox("Selecionar modelo", df['model'].unique())

df_laptops_modelo = df[df["model"] == model]

df_laptops_final = df[df['cluster'] == df_laptops_modelo.iloc[0]['cluster']]

st.write("Recomendações de Modelos")
st.table(df_laptops_final)