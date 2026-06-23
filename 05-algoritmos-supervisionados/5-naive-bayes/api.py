from flask import Flask, jsonify
from pydantic import BaseModel
from flask_pydantic import validate
import joblib
from sklearn.naive_bayes import GaussianNB
import pandas as pd

app = Flask(__name__)

class request_body(BaseModel):
    Genero_Masculino: int
    Idade: int
    Historico_Familiar_Sobrepeso: int
    Consumo_Alta_Caloria_Com_Frequencia: int
    Consumo_Vegetais_Com_Frequencia: int
    Refeicoes_Dia: int
    Consumo_Alimentos_entre_Refeicoes: int
    Fumante: int
    Consumo_Agua: int
    Monitora_Calorias_Ingeridas: int
    Nivel_Atividade_Fisica: int
    Nivel_Uso_Tela: int
    Consumo_Alcool: int
    Transporte_Automovel: int
    Transporte_Bicicleta: int
    Transporte_Motocicleta: int
    Transporte_Publico: int
    Transporte_Caminhada: int

modelo_obesidade: GaussianNB = joblib.load("./modelo_obesidade.pkl")

@app.route("/predict", methods=['POST'])
@validate()
def predict(body: request_body):
    df_predict = pd.DataFrame(body.model_dump(), index=[1])

    bins = [x * 10 for x in range(1,8)]
    bins_ordinal = list(range(0,6))
    df_predict["Faixa_Etaria"] = pd.cut(
      x=df_predict['Idade'],
      bins=bins,
      labels=bins_ordinal,
      include_lowest=True
    )

    # deixar as 8 melhores features
    df_predict = df_predict[[
        'Historico_Familiar_Sobrepeso',
        'Consumo_Alta_Caloria_Com_Frequencia',
        'Consumo_Alimentos_entre_Refeicoes',
        'Monitora_Calorias_Ingeridas',
        'Nivel_Atividade_Fisica',
        'Nivel_Uso_Tela',
        'Transporte_Caminhada',
        'Faixa_Etaria'
    ]]

    y_pred = modelo_obesidade.predict(df_predict)[0].astype(int)

    return jsonify({
        'obesity': y_pred.tolist()
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)