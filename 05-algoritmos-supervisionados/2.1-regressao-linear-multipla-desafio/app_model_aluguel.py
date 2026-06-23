import gradio as gr
import joblib
import pandas as pd

modelo = joblib.load("./modelo_aluguel.pkl")

def predict(metros_quadrados, n_quartos, idade, garagem, periferia, suburbio):
    dict_predict = {
        'metros_quadrados': float(metros_quadrados),
        'n_quartos': int(n_quartos),
        'idade': float(idade),
        'garagem': int(garagem),
        'periferia': int(periferia),
        'suburbio': int(suburbio),
    }

    predict_df = pd.DataFrame(dict_predict, index=[0])
    valor_aluguel = modelo.predict(predict_df)
    return float(valor_aluguel[0])

demo = gr.Interface(
    fn=predict,
    inputs=[
       gr.Slider(50, 198, step=1),
       gr.Slider(1, 5, step=1),
       gr.Slider(0, 50, step=1),
       gr.Radio([False, True], label="Garagem (False = Não, True = Sim)"),
       gr.Radio([False, True], label="Periferia (False = Não, True = Sim)"),
       gr.Radio([False, True], label="Subúrbio (False = Não, True = Sim)"),
    ],
    outputs="number",
)

demo.launch()