import os
from flask import Flask, request, jsonify
from joblib import load
from flasgger import Swagger

app = Flask(__name__)

# Swagger config
swagger = Swagger(app, template={
    "info": {
        "title": "Wine Quality API",
        "description": "API de predição usando modelo ML",
        "version": "1.0.0"
    }
})

# Caminho do modelo
modelo_path = os.path.join('models', 'modelo.pkl')

# Carregar modelo
modelo = load(modelo_path)

# Ordem fixa das features (CRÍTICO)
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "color"
]

# Exemplo padrão
EXEMPLO = {
    "fixed acidity": 7.0,
    "volatile acidity": 0.27,
    "citric acid": 0.36,
    "residual sugar": 20.7,
    "chlorides": 0.045,
    "free sulfur dioxide": 45.0,
    "total sulfur dioxide": 170.0,
    "density": 1.0010,
    "pH": 3.00,
    "sulphates": 0.45,
    "alcohol": 8.8,
    "color": 1
}

def preparar_entrada(data):
    try:
        return [data[f] for f in FEATURES]
    except KeyError as e:
        raise ValueError(f"Campo ausente: {str(e)}")

def mapear_resultado(pred):
    return ['ruim' if p == 1 else 'bom' for p in pred]


@app.route('/example', methods=['GET'])
def example():
    """
    Exemplo de predição
    ---
    responses:
      200:
        description: Retorna uma predição de exemplo
    """
    entrada = preparar_entrada(EXEMPLO)
    previsao = modelo.predict([entrada])
    return jsonify({
        "input": EXEMPLO,
        "prediction": mapear_resultado(previsao)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predição com modelo ML
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            fixed acidity: {type: number}
            volatile acidity: {type: number}
            citric acid: {type: number}
            residual sugar: {type: number}
            chlorides: {type: number}
            free sulfur dioxide: {type: number}
            total sulfur dioxide: {type: number}
            density: {type: number}
            pH: {type: number}
            sulphates: {type: number}
            alcohol: {type: number}
            color: {type: integer}
    responses:
      200:
        description: Resultado da predição
      400:
        description: Erro de validação
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON inválido"}), 400

        entrada = preparar_entrada(data)
        predicao = modelo.predict([entrada])
        resultado = mapear_resultado(predicao)

        return jsonify({
            "prediction": resultado
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        return jsonify({"error": "Erro interno"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
