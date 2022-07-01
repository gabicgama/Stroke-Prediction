import xgboost as xgb
from flask import Flask, request
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger
import pickle
import os

app = Flask(__name__)

swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "Predição de AVC",
        "description": "API desenvolvida por Gabriela Campos Gama para o trabalho de conclusão do curso de Engenharia da Computaçao - CEFET/MG",
        "contact": {
            "responsibleOrganization": "ME",
            "responsibleDeveloper": "Me",
            "email": "camposgamagabriela@gmail.com",
        },
        "termsOfService": "http://me.com/terms",
        "version": "0.0.1"
    }
}

Swagger(app, config=swagger_config, template=template)
# Adaboost
pickle_in = open('models/model_ab.pkl', 'rb')
model_ab = pickle.load(pickle_in)
# Extra Trees
pickle_in = open('models/model_et.pkl', 'rb')
model_et = pickle.load(pickle_in)
# Gradient Boosting
pickle_in = open('models/model_gbc.pkl', 'rb')
model_gbc = pickle.load(pickle_in)
# GaussinNB
pickle_in = open('models/model_gnb.pkl', 'rb')
model_gnb = pickle.load(pickle_in)
# KNN
pickle_in = open('models/model_knn.pkl', 'rb')
model_knn = pickle.load(pickle_in)
# Random Forest
pickle_in = open('models/model_rf.pkl', 'rb')
model_rf = pickle.load(pickle_in)
# XGBoost
pickle_in = open('models/model_xgb.pkl', 'rb')
model_xgb = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return 'Deploy do modelo de ML'


@app.route('/predicao_parametros', methods=["Get"])
def predict_stroke_parameters():
    """Predição utilizando os parâmetros
    ---
    tags:
          - Predição
    parameters:
        - name: idade
          in: query
          type: number
          required: true
        - name: genero
          in: query
          type: number
          required: true
        - name: hipertensao
          in: query
          type: number
          required: true
        - name: doenca_do_coracao
          in: query
          type: number
          required: true
        - name: ja_se_casou
          in: query
          type: number
          required: true
        - name: tipo_trabalho
          in: query
          type: number
          required: true
        - name: tipo_residencia
          in: query
          type: number
          required: true
        - name: nivel_glicose
          in: query
          type: number
          required: true
        - name: imc
          in: query
          type: number
          required: true
        - name: condicao_fumante
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    """

    age = request.args.get("idade")
    gender = request.args.get("genero")
    hypertension = request.args.get("hipertensao")
    heart_disease = request.args.get("doenca_do_coracao")
    ever_married = request.args.get("ja_se_casou")
    work_type = request.args.get("tipo_trabalho")
    residence_type = request.args.get("tipo_residencia")
    avg_glucose_level = request.args.get("nivel_glicose")
    bmi = request.args.get("imc")
    smoking_status = request.args.get("condicao_fumante")

    predictions = [
        model_ab.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]]),
        model_et.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]]),
        model_gbc.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]]),
        model_gnb.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]]),
        model_knn.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]]),
        model_rf.predict([[hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type,
                           residence_type, smoking_status, age]])]

    predictions_concat = np.concatenate(predictions)
    print(predictions_concat)
    result = model_xgb.predict([predictions_concat])

    return str(result[0])


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, host='0.0.0.0',
            port=int(os.environ.get('PORT', 5080)))
