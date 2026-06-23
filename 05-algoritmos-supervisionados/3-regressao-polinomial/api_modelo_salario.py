from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

app = FastAPI()

class RequestBody(BaseModel):
    tempo_na_empresa: int
    nivel_na_empresa: int

model_poly: LinearRegression = joblib.load("./modelo_poly_salario.pkl")

@app.post("/get-salary")
def predictSalary(request: RequestBody):
    input_features = {
        'tempo_na_empresa': request.tempo_na_empresa,
        'nivel_na_empresa': request.nivel_na_empresa
    }

    pred_df = pd.DataFrame(input_features, index=[1])
    
    y_pred = model_poly.predict(pred_df)[0].astype(float)

    return { 
        'predicted_salary': y_pred
    }