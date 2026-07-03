import joblib
import gradio as gr
import pandas as pd

modelo = joblib.load('./clustering_model_km.pkl')
preprocessor = joblib.load("./clustering_pipeline.pkl")

def clustering(arquivo):
    df_empresas = pd.read_csv("./dataset/clientes.csv")

    X_transformed = preprocessor.fit_transform(df_empresas)

    modelo.fit(X_transformed)

    df_empresas['cluster'] = modelo.labels_
    df_empresas.to_csv('./dataset/clusters.csv', index=False)

    return './dataset/clusters.csv'

app = gr.Interface(
    clustering,
    gr.File(file_types=['.csv']),
    "file",
)

app.launch()
