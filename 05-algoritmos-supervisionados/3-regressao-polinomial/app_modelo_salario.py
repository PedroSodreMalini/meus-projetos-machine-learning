import streamlit as st
import json
import requests

# Título da aplicação
st.title("Modelo de Predição de Salário")

# Inputs do Usuário
st.write("Quantos meses o profissional está na empresa?")
tempo_na_empresa = st.slider("Meses", min_value=1, max_value=120, value=60, step=1 )

# Inputs do Usuário
st.write("Qual o nível do profissional empresa?")
nivel_na_empresa = st.slider("Nível", min_value=1, max_value=10, value=5, step=1 )

# Prepara dados para a API
input_features = { 
    'tempo_na_empresa': tempo_na_empresa,
    'nivel_na_empresa': nivel_na_empresa, 
    }

# Criar um botão e capturar um evento para este botão
if st.button('Estimar Salário'):
    res = requests.post(
        "http://localhost:8000/get-salary",
        json=input_features
    )
    res_json = json.loads(res.text)
    salario_em_reais = round(res_json["predicted_salary"], 2)
    st.subheader(f"O salário estimado é de {salario_em_reais}.")