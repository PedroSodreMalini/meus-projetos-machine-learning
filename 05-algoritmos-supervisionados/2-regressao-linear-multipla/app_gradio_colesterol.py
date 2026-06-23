import gradio as gr
import joblib
import pandas as pd

modelo = joblib.load("./modelo_colesterol.pkl")

def predict(grupo_sanguineo,  nivel_atividade_fisica, fumante, idade, altura, peso):
    predicao_individual = {
        'grupo_sanguineo': grupo_sanguineo,
        'fumante': fumante,
        'nivel_atividade_fisica': nivel_atividade_fisica,
        'idade': idade,
        'peso': peso,
        'altura': altura
    }
    predict_df = pd.DataFrame(predicao_individual, index=[1])
    colesterol = modelo.predict(predict_df)
    return colesterol[0]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(['O', 'A', 'B', 'AB']),
        gr.Radio(['Baixo', 'Moderado', 'Alto']),
        gr.Radio(['Sim', 'Não']),
        gr.Slider(20, 80, step=1),
        gr.Slider(150, 200, step=1),
        gr.Slider(40, 160, step=0.1),
    ],
    outputs=[
        'number'
    ],
)

demo.launch()