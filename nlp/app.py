import os
import re

from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# Evita warning chato do tokenizer em ambiente containerizado
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

swagger = Swagger(app, template={
    "info": {
        "title": "NLP API",
        "description": "API para NLP: Análise de sentimentos com Transformers e vetorização TF-IDF.",
        "version": "1.0.0"
    }
})


# Modelo padrão de sentiment-analysis:
# distilbert-base-uncased-finetuned-sst-2-english

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    framework="tf"
)

STOPWORDS_PT = {
    "a", "o", "as", "os",
    "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem",
    "e", "ou", "mas",
    "que", "se", "ao", "aos",
    "é", "foi", "são", "ser", "estar",
    "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas",
    "isso", "isto", "aquele", "aquela",
    "pelo", "pela", "pelos", "pelas",
    "lhe", "eles", "elas", "ele", "ela",
    "meu", "minha", "seu", "sua",
    "dos", "das", "nos", "nas"
}


def preprocess_text(text):
    """
    Pré-processamento simples:
    - minúsculas
    - remove pontuação
    - remove stopwords básicas em português
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    tokens = [
        token for token in tokens
        if token not in STOPWORDS_PT and len(token) > 1
    ]
    return " ".join(tokens)

@app.route("/", methods=["GET"])
def index():
    """
    Página inicial da API
    ---
    responses:
      200:
        description: Status da API
    """
    return jsonify({
        "status": "ok",
        "service": "NLP API",
        "routes": {
            "health": "GET /health",
            "sentiment": "POST /analyze_sentiment",
            "vectorize": "POST /vectorize",
            "docs": "GET /apidocs"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check
    ---
    responses:
      200:
        description: API disponível
    """
    return jsonify({
        "status": "healthy"
    })


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    """
    Análise de sentimento com Transformers
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              example: "This class is excellent and very useful"
    responses:
      200:
        description: Resultado da análise de sentimento
      400:
        description: Erro de validação
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({
            "error": "Envie um JSON com o campo 'text'"
        }), 400

    text = data["text"]

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "O campo 'text' deve ser uma string não vazia"
        }), 400

    result = sentiment_pipeline(text)

    return jsonify({
        "input": text,
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "result": result
    })


@app.route("/vectorize", methods=["POST"])
def vectorize():
    """
    Vetorização de textos com TF-IDF
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - texts
          properties:
            texts:
              type: array
              items:
                type: string
              example:
                - "o juiz julgou o processo"
                - "o magistrado analisou os autos"
                - "o processo foi analisado pelo juiz"
    responses:
      200:
        description: Vocabulário e matriz TF-IDF
      400:
        description: Erro de validação
    """
    data = request.get_json(silent=True)

    if not data or "texts" not in data:
        return jsonify({
            "error": "Envie um JSON com o campo 'texts' contendo uma lista de textos"
        }), 400

    texts = data["texts"]

    if not isinstance(texts, list) or not texts:
        return jsonify({
            "error": "O campo 'texts' deve ser uma lista não vazia"
        }), 400

    if not all(isinstance(text, str) and text.strip() for text in texts):
        return jsonify({
            "error": "Todos os itens de 'texts' devem ser strings não vazias"
        }), 400

    processed_texts = [preprocess_text(text) for text in texts]

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(processed_texts)
    except ValueError as exc:
        return jsonify({
            "error": "Não foi possível vetorizar os textos",
            "detail": str(exc)
        }), 400

    return jsonify({
        "original_texts": texts,
        "processed_texts": processed_texts,
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "tfidf_vectors": matrix.toarray().tolist()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Rota não encontrada"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Erro interno da aplicação"
    }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )