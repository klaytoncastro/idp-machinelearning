import os
from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Obter o caminho completo para o arquivo modelo.pkl na pasta models
modelo_path = os.path.join('models', 'modelo.pkl')

# Carregar o modelo
modelo = load(modelo_path)

# Definir rota para receber requisições POST
@app.route('/predict', methods=['POST'])
def predict():
    # Receber dados JSON da requisição
    data = request.get_json()
    
    # Fazer a previsão usando o modelo
    predicao = modelo.predict(data)
    
    # Mapear o resultado da previsão para uma resposta legível
    resultado = ['ruim' if pred == 0 else 'bom' for pred in predicao]
    
    # Retornar o resultado como JSON
    return jsonify(resultado)

# Executar o aplicativo Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


'''from flask import Flask
import redis

app = Flask(__name__)
db = redis.Redis(host='redis', port=6379)

# make redis
#redis_cache = redis.Redis(host='localhost', port=6379, db=0, password="redis_password")

@app.route('/')
def hello():
    count = db.incr('hits')
    return 'Hello World! I have been seen {} times.\n'.format(count)

@app.route('/sem-redis')
def hellow():
    #count = db.incr('hits')
    return 'Hello World! I have been seen times.'

@app.route('/set/<string:key>/<string:value>')
def set(key, value):
    if db.exists(key):
        pass
    else:
        db.set(key, value)
    return "OK"

@app.route('/get/<string:key>')
def get(key):
    if db.exists(key):
        return db.get(key)
    else:
        return f"{key} is not exists"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
'''