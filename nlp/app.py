import os
import re
from functools import lru_cache

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, jsonify, request
from flasgger import Swagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "NLP API",
        "description": (
            "API educacional de NLP: análise de sentimentos com DistilBERT, "
            "TF-IDF, embeddings multilíngues e similaridade por cosseno. "
            "O TF-IDF usa um pré-processamento básico em português; "
            "os embeddings recebem os textos originais."
        ),
        "version": "1.1.0"
    },
    "tags": [
        {"name": "Sistema", "description": "Informações e disponibilidade da API"},
        {"name": "Sentimentos", "description": "Classificação com DistilBERT (modelo ajustado para inglês)"},
        {"name": "TF-IDF", "description": "Representação estatística de documentos"},
        {"name": "Embeddings", "description": "Representação semântica multilíngue"},
        {"name": "Similaridade", "description": "Comparação de documentos por cosseno"}
    ]
})

# O modelo de sentimentos continua carregado na inicialização, como no código original.
sentiment_pipeline = pipeline(
    task="sentiment-analysis", model=MODEL_NAME, framework="pt", device=-1
)

# Embeddings carregados apenas na primeira requisição que precisa deles.
@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")

STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "e", "ou", "mas", "que", "se",
    "ao", "aos", "é", "foi", "são", "ser", "estar", "este", "esta",
    "estes", "estas", "esse", "essa", "esses", "essas", "isso",
    "isto", "aquele", "aquela", "pelo", "pela", "pelos", "pelas",
    "lhe", "eles", "elas", "ele", "ela", "meu", "minha", "seu", "sua"
}


def preprocess_text(text):
    """Normaliza o texto para TF-IDF; não é utilizado nos embeddings."""
    tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    return " ".join(token for token in tokens
                    if token not in STOPWORDS_PT and len(token) > 1)


def validate_texts(data, min_items=1):
    if not isinstance(data, dict) or "texts" not in data:
        return None, "Envie um JSON com o campo 'texts' contendo uma lista de textos"
    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) < min_items:
        return None, f"O campo 'texts' deve ser uma lista com pelo menos {min_items} texto(s)"
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return None, "Todos os itens de 'texts' devem ser strings não vazias"
    return texts, None


def tfidf_for(texts):
    processed_texts = [preprocess_text(t) for t in texts]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_texts)
    return processed_texts, vectorizer, matrix


def embeddings_for(texts):
    return get_embedding_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False
    )


@app.route("/", methods=["GET"])
def index():
    """Informações da API e rotas disponíveis
    ---
    tags:
      - Sistema
    produces:
      - application/json
    responses:
      200:
        description: Versão, status e rotas disponíveis
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            service: {type: string, example: NLP API}
            version: {type: string, example: '1.1.0'}
            routes: {type: object}
    """
    return jsonify({
        "status": "ok", "service": "NLP API", "version": "1.1.0",
        "routes": {
            "health": "GET /health", "sentiment": "POST /analyze_sentiment",
            "vectorize": "POST /vectorize", "tfidf_similarity": "POST /similarity/tfidf",
            "embeddings": "POST /embeddings",
            "semantic_similarity": "POST /similarity/semantic",
            "docs": "GET /apidocs"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check básico
    ---
    tags:
      - Sistema
    produces:
      - application/json
    responses:
      200:
        description: Confirma que a aplicação HTTP está respondendo (não testa os modelos)
        schema:
          type: object
          properties:
            status: {type: string, example: healthy}
    """
    return jsonify({"status": "healthy"})


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    """Análise de sentimentos com DistilBERT
    O modelo foi ajustado para classificação positiva/negativa em inglês.
    O score é uma saída do classificador, não uma garantia de acerto.
    ---
    tags:
      - Sentimentos
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: Frase a ser classificada
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              example: This class is excellent and very useful
    responses:
      200:
        description: Sentimento previsto, score e identificação do modelo
        schema:
          type: object
          properties:
            input: {type: string, example: This class is excellent and very useful}
            model: {type: string, example: distilbert-base-uncased-finetuned-sst-2-english}
            result:
              type: array
              items:
                type: object
                properties:
                  label: {type: string, example: POSITIVE}
                  score: {type: number, format: float, example: 0.9999}
      400:
        description: JSON ausente ou campo text inválido
        schema:
          type: object
          properties:
            error: {type: string, example: "O campo 'text' deve ser uma string não vazia"}
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Envie um JSON com o campo 'text'"}), 400
    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "O campo 'text' deve ser uma string não vazia"}), 400
    result = sentiment_pipeline(text, truncation=True)
    return jsonify({"input": text, "model": MODEL_NAME, "result": result})


@app.route("/vectorize", methods=["POST"])
def vectorize():
    """Vetorização TF-IDF de documentos
    Retorna textos originais, pré-processamento, vocabulário e matriz.
    Cada linha da matriz representa um documento e cada coluna, um termo.
    ---
    tags:
      - TF-IDF
    consumes:
      - application/json
    produces:
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
              items: {type: string}
              example:
                - o juiz julgou o processo
                - o magistrado analisou os autos
                - o processo foi analisado pelo juiz
    responses:
      200:
        description: Vocabulário e matriz TF-IDF (documentos x termos)
        schema:
          type: object
          properties:
            original_texts:
              type: array
              items: {type: string}
            processed_texts:
              type: array
              items: {type: string}
            vocabulary:
              type: array
              items: {type: string}
              example: [analisado, analisou, autos, juiz, julgou, magistrado, processo]
            tfidf_vectors:
              type: array
              items:
                type: array
                items: {type: number, format: float}
      400:
        description: Lista inválida ou vocabulário vazio após pré-processamento
        schema:
          type: object
          properties:
            error: {type: string}
            detail: {type: string}
    """
    data = request.get_json(silent=True)
    texts, error = validate_texts(data)
    if error:
        return jsonify({"error": error}), 400
    try:
        processed, vectorizer, matrix = tfidf_for(texts)
    except ValueError as exc:
        return jsonify({"error": "Não foi possível vetorizar os textos",
                        "detail": str(exc)}), 400
    return jsonify({
        "original_texts": texts,
        "processed_texts": processed,
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "tfidf_vectors": matrix.toarray().tolist()
    })


@app.route("/similarity/tfidf", methods=["POST"])
def similarity_tfidf():
    """Similaridade lexical por cosseno sobre vetores TF-IDF
    A matriz N x N compara cada documento com todos os outros.
    ---
    tags:
      - Similaridade
    consumes:
      - application/json
    produces:
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
              minItems: 2
              items: {type: string}
              example:
                - O juiz analisou o processo
                - O magistrado examinou os autos
                - O restaurante oferece comida italiana
    responses:
      200:
        description: Matriz de similaridade lexical N x N
        schema:
          type: object
          properties:
            method: {type: string, example: tfidf_cosine}
            texts:
              type: array
              items: {type: string}
            processed_texts:
              type: array
              items: {type: string}
            vocabulary:
              type: array
              items: {type: string}
            similarity_matrix:
              type: array
              items:
                type: array
                items: {type: number, format: float}
      400:
        description: São necessários pelo menos dois textos válidos
        schema:
          type: object
          properties:
            error: {type: string}
    """
    data = request.get_json(silent=True)
    texts, error = validate_texts(data, min_items=2)
    if error:
        return jsonify({"error": error}), 400
    try:
        processed, vectorizer, matrix = tfidf_for(texts)
    except ValueError as exc:
        return jsonify({"error": "Não foi possível calcular TF-IDF",
                        "detail": str(exc)}), 400
    return jsonify({
        "method": "tfidf_cosine", "texts": texts,
        "processed_texts": processed,
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "similarity_matrix": cosine_similarity(matrix).tolist()
    })


@app.route("/embeddings", methods=["POST"])
def embeddings():
    """Gera embeddings semânticos multilíngues
    Cada frase é representada por 384 números. Diferentemente de TF-IDF,
    o modelo recebe as frases originais, sem remoção de stopwords.
    ---
    tags:
      - Embeddings
    consumes:
      - application/json
    produces:
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
              items: {type: string}
              example:
                - O juiz analisou o processo
                - O magistrado examinou os autos
    responses:
      200:
        description: Embeddings L2-normalizados, um vetor de 384 posições por texto
        schema:
          type: object
          properties:
            method: {type: string, example: sentence_transformers}
            model: {type: string, example: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}
            dimensions: {type: integer, example: 384}
            texts:
              type: array
              items: {type: string}
            embeddings:
              type: array
              items:
                type: array
                items: {type: number, format: float}
      400:
        description: Lista de textos inválida
        schema:
          type: object
          properties:
            error: {type: string}
      500:
        description: Falha interna durante a inferência ou carregamento do modelo
    """
    data = request.get_json(silent=True)
    texts, error = validate_texts(data)
    if error:
        return jsonify({"error": error}), 400
    vectors = embeddings_for(texts)
    return jsonify({
        "method": "sentence_transformers", "model": EMBEDDING_MODEL,
        "dimensions": int(vectors.shape[1]), "texts": texts,
        "embeddings": vectors.tolist()
    })


@app.route("/similarity/semantic", methods=["POST"])
def similarity_semantic():
    """Similaridade semântica por cosseno entre embeddings
    Usa um Transformer multilíngue; não compara somente palavras iguais.
    O cosseno pode variar de -1 a 1, embora valores reais dependam dos textos.
    ---
    tags:
      - Similaridade
    consumes:
      - application/json
    produces:
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
              minItems: 2
              items: {type: string}
              example:
                - O juiz analisou o processo
                - O magistrado examinou os autos
                - O restaurante oferece comida italiana
    responses:
      200:
        description: Matriz de similaridade semântica N x N
        schema:
          type: object
          properties:
            method: {type: string, example: semantic_cosine}
            model: {type: string, example: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}
            dimensions: {type: integer, example: 384}
            texts:
              type: array
              items: {type: string}
            similarity_matrix:
              type: array
              items:
                type: array
                items: {type: number, format: float}
      400:
        description: São necessários pelo menos dois textos válidos
        schema:
          type: object
          properties:
            error: {type: string}
      500:
        description: Falha interna durante a inferência ou carregamento do modelo
    """
    data = request.get_json(silent=True)
    texts, error = validate_texts(data, min_items=2)
    if error:
        return jsonify({"error": error}), 400
    vectors = embeddings_for(texts)
    return jsonify({
        "method": "semantic_cosine", "model": EMBEDDING_MODEL,
        "dimensions": int(vectors.shape[1]), "texts": texts,
        "similarity_matrix": cosine_similarity(vectors).tolist()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Rota não encontrada"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Método HTTP não permitido"}), 405


@app.errorhandler(500)
def internal_error(error):
    app.logger.exception("Erro interno ao processar uma requisição")
    return jsonify({"error": "Erro interno da aplicação"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
