from flask import Flask, jsonify
from flask_pydantic import validate
import pandas as pd
from pydantic import BaseModel
from sklearn.naive_bayes import GaussianNB
import joblib

app = Flask(__name__)

class request_body(BaseModel):
    glicemia: int
    pressao_arterial: int

model: GaussianNB = joblib.load("./modelo_diabetes.pkl")

@app.route("/predict", methods=['POST'])
@validate()
def predict(body: request_body):
    df_predict = pd.DataFrame(body.model_dump(), index=[1])

    y_pred = model.predict(df_predict)[0].astype(int)

    return jsonify({
      'diabetes': y_pred.tolist()
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)