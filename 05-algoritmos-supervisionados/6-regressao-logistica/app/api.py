from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("./model_lr.pkl")
# obs: dados devem vir normalizados
class request_body(BaseModel):
    A_id: int
    Size: float
    Weight: float
    Sweetness: float
    Crunchiness: float
    Juiciness: float
    Ripeness: float
    Acidity: float

@app.post("/classify")
def predict(data: request_body):
    input_features = [[
      data.Size,
      data.Weight,
      data.Sweetness,
      data.Crunchiness,
      data.Juiciness,
      data.Ripeness,
      data.Acidity
    ]]

    y_pred = model.predict(input_features)[0].astype(int)
    y_prob = model.predict_proba(input_features)[0].astype(float)

    resposta = 'Boa' if y_pred == 1 else "Ruim"
    probabilidade = y_prob[y_pred]

    return {
        "qualidade": resposta,
        "probabilidade_de_ter_qualidade_acima": probabilidade,
    }
