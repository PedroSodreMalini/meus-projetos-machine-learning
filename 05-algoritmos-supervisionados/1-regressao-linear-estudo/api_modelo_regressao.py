from pydantic import BaseModel
from fastapi import FastAPI
from sys import exit as leave
import joblib
from sklearn.linear_model import LinearRegression

app = FastAPI()

class PredictPointsRequest(BaseModel):
    horas_estudo: float

class PredictHoursRequest(BaseModel):
    pontos: int

modelo: LinearRegression = joblib.load("./modelo_regressao.pkl")
if not isinstance(modelo, LinearRegression):
    print("Modelo não é de Regressão Linear")
    leave()

@app.post('/predict-points')
def predict_points(data: PredictPointsRequest):
    input_feature = [[data.horas_estudo]]
    y_pred= modelo.predict(input_feature)[0][0].astype(int)
    return {'pontuacao_prevista': y_pred.tolist()}

@app.post('/predict-hours')
def predict_hours(data: PredictHoursRequest):
    horas_prev: float = ((data.pontos - modelo.intercept_[0])/modelo.coef_[0][0])
    return {'horas_previstas': round(horas_prev, 2)}
