# NLP (Processamento de Linguagem Natural)

## 1. Visão Geral

NLP (Natural Language Processing), ou Processamento de Linguagem Natural, é uma área da Inteligência Artificial que abrange as tarefas relacionadas ao entendimento e à geração de texto em linguagem humana. Por meio de NLP, máquinas executam tarefas de interpretação, análise e resposta de textos de maneira inteligente, facilitando a interação humano-computador. 

Dentre as tarefas comuns de NLP, podemos citar análise de sentimentos, classificação de texto, resposta a perguntas e tradução automática. Nesse cenário, os LLMs (Large Language Models), como GPT-4, LLaMA e Claude, são modelos de NLP em grande escala, com bilhões de parâmetros, valores ajustáveis que uma RNA captura durante o processo de treinamento. Esses modelos avançados têm uma capacidade extraordinária de análise e geração texto, que tem revolcuinado as possibilidades de criar aplicações cada vez mais sofisticadas em NLP. 

## 2. Principais Tarefas em NLP

Aqui estão algumas tarefas relevantes comumente aplicadas em NLP:

### 2.1. Análise de Sentimentos

A análise de sentimentos é uma tarefa de NLP que identifica a emoção ou polaridade de um texto, classificando-o como positivo, negativo ou neutro. É usada para entender a percepção dos usuários em redes sociais, avaliações de produtos e feedbacks de clientes, entre outros.

### 2.2. Classificação de Texto

A classificação de texto envolve categorizar frases, parágrafos ou documentos inteiros em categorias específicas. Por exemplo, classificar e-mails como “Spam” ou “Não Spam” ou agrupar notícias em categorias como “Esportes”, “Política” e “Tecnologia”.

### 2.3. Resposta a Perguntas

Nesta tarefa, o modelo recebe uma pergunta e um contexto, e deve responder com base nas informações fornecidas. Isso é útil em assistentes virtuais e chatbots que precisam responder a consultas específicas dos usuários com precisão.

### 2.4. Geração de Texto

A geração de texto permite que um modelo complete frases, redija parágrafos ou crie conteúdo original a partir de uma entrada inicial. Usada para gerar artigos, auxiliar em escrita criativa ou responder a mensagens de maneira personalizada.

### 2.5. Reconhecimento de Entidades Nomeadas (NER)

O reconhecimento de entidades nomeadas (NER) identifica e classifica entidades específicas em um texto, como nomes de pessoas, locais, organizações, datas e quantidades. Isso é valioso em análise de documentos, mineração de dados e na extração de informações estruturadas de textos não estruturados.

## 3. Componentes de Processamento em NLP

O processamento de linguagem natural é dividido em diferentes componentes, ou camadas de análise, que permitem transformar o texto em representações estruturadas e compreensíveis para sistemas computacionais. As seguintes etapas são fundamentais para a construção de modelos NLP mais precisos:

### 3.1 Análise Léxica

A análise léxica identifica as palavras no texto e as transforma em `tokens`, que são as unidades mínimas de significado. O processo de tokenização, que pode ser realizado em Python com a biblioteca `ntlk`, é uma técnica essencial nesta etapa, convertendo o texto em uma sequência de palavras ou frases (`tokens`).

### 3.2 Análise Sintática 

Examina a estrutura gramatical das frases, determinando as relações entre palavras. Esta fase é responsável por identificar a estrutura da sentença, como sujeito, predicado e objetos, o que ajuda a entender a estrutura gramatical do texto.

### 3.3 Análise Semântica 

Focada no significado das palavras e frases, essa análise interpreta as intenções subjacentes no texto. Em NLP, isso pode envolver o uso de embeddings para representar o significado das palavras em um espaço vetorial, auxiliando em tarefas como desambiguação de sentidos.

### 3.4 Análise do Discurso

Considera as relações entre sentenças e o contexto geral do texto, essencial para tarefas de sumarização e extração de tópicos, onde a continuidade e o significado global do texto são importantes.

### 3.5 Análise Pragmática 

Este componente vai além da estrutura e significado básico, interpretando a intenção e o contexto situacional. É especialmente importante em aplicações como chatbots, onde o entendimento da intenção do usuário é crucial.

## 4. Mecanismos Computacionais em NLP

Para realizar tarefas de NLP, várias técnicas e modelos computacionais são empregados. As principais abordagens incluem:

<!-- redes neurais profundas, embeddings, transformers e pipelines de processamento, que tornam possível a análise e interpretação de grandes volumes de texto. -->

### 4.1 Redes Neurais Profundas (Deep Learning)

As redes neurais desempenham um papel crucial em NLP, particularmente em tarefas complexas de entendimento e geração de texto. Aqui estão algumas das redes neurais mais utilizadas:

### 4.2 Redes Neurais Recorrentes (RNNs)

As RNNs, especialmente as LSTMs (Long Short-Term Memory) e GRUs (Gated Recurrent Units), são redes que mantêm uma memória de estados anteriores, o que é valioso para sequências de dados, como frases e parágrafos. Elas são eficazes para modelar dependências de curto e médio prazo entre palavras e são usadas em tarefas como tradução automática e geração de texto.

### 4.3 Transformers 

O advento dos transformers revolucionou o NLP. Ao contrário das RNNs, os transformers processam o texto em paralelo, usando mecanismos de self-attention para identificar e ponderar as relações entre todas as palavras de uma sequência, independentemente da posição delas. Modelos como BERT (Bidirectional Encoder Representations from Transformers) e GPT (Generative Pre-trained Transformer) são baseados em transformers e conseguem capturar tanto o contexto local quanto o global, gerando representações mais ricas e precisas dos textos.

### 4.4 Embeddings de Palavras

Os embeddings são representações vetoriais de palavras que capturam o significado semântico delas. Em vez de representar palavras por frequências ou características sintáticas, os embeddings posicionam palavras com significados similares em regiões próximas de um espaço vetorial de alta dimensionalidade.

- **Word2Vec** e **GloVe**: São duas técnicas populares para gerar embeddings. O Word2Vec utiliza uma rede neural rasa para aprender representações das palavras com base no contexto em que elas aparecem. O GloVe (Global Vectors for Word Representation) usa uma abordagem baseada em matriz de coocorrência, gerando embeddings que capturam relações globais entre as palavras.

### 4.5 Embeddings Contextuais (BERT, ELMo) 

Modelos como BERT e ELMo (Embeddings from Language Models) geram embeddings dependentes do contexto, onde a representação de uma palavra varia conforme a frase em que ela está inserida. Isso melhora a precisão em tarefas que exigem entendimento contextual.

## 5. Técnicas de Pré-Processamento e Vetorização

Antes de alimentar o texto em modelos de NLP, é necessário transformá-lo em uma representação numérica. As principais técnicas incluem:

- **Bag of Words (BoW)** e **TF-IDF**: Essas técnicas criam matrizes de termos em que cada documento é representado pela frequência das palavras (BoW) ou pela frequência ajustada por raridade (TF-IDF). Essas abordagens são computacionalmente simples e foram amplamente usadas antes dos embeddings de palavras.

- **Tokenização** e **Normalização**: A tokenização divide o texto em unidades menores, como palavras ou frases. A normalização envolve tarefas como transformar palavras em letras minúsculas, remoção de stopwords (palavras comuns sem valor semântico significativo) e stemming (redução das palavras às suas raízes).

## 6. Atividade Prática 

### Tarefa 1: Análise de Sentimentos com Hugging Face Transformers

Nesta tarefa, usaremos a biblioteca Hugging Face Transformers para realizar análise de sentimentos em texto. O modelo pré-treinado de análise de sentimentos permite classificar um texto como "positivo", "negativo" ou "neutro". Utilizem como base a infraestrutura que implementamos com Flask em [Production](https://github.com/klaytoncastro/idp-machinelearning/tree/main/production). Implementem uma nova rota na aplicação para avaliar textos utilizando um modelo de classificação de sentimentos pré-treinado da [Hugging Face](https://huggingface.co/blog/sentiment-analysis-python). 

Vocês deverão modificar o código existente para carregar o modelo de análise de sentimentos e criar uma rota `/analyze_sentiment` que aceite requisições `POST` com um `JSON` contendo um texto e retorne o sentimento classificado como 'positivo', 'negativo' ou 'neutro'. Essa rota deve aceitar requisições POST com um JSON contendo o texto a ser analisado, conforme exemplo abaixo: 

```json
{ "text": "Seu texto aqui" }
```

Testem a nova funcionalidade utilizando a ferramenta `curl`, similares aos mostrados anteriormente na atividade [Production](https://github.com/klaytoncastro/idp-machinelearning/tree/main/production): 

```bash
curl -X POST http://localhost:5000/analyze_sentiment \
-H "Content-Type: application/json" \
-d '{"text": "Seu texto aqui"}'
```

<!--
Classificação de Sentimento: Use a pipeline de sentiment-analysis da Hugging Face para processar o texto e retornar o sentimento classificado.

Formato de Entrada e Saída: O JSON de entrada deverá conter uma chave "text" com o texto a ser analisado. O JSON de saída deve conter a classificação de sentimento.


```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/', methods=['GET'])
def index():
    return 'O Flask está funcionando!'

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Receber dados JSON da requisição
    data = request.get_json()

    # Extrair os valores do JSON
    text = data['text']

    # Fazer a previsão do sentimento usando o modelo
   
    resultado = sentiment_pipeline(text)

    # Retornar o resultado como JSON
    return jsonify(resultado)

# Executar o aplicativo Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```
--> 

### Tarefa 2: Vetorização com NLTK e TF-IDF

Nesta tarefa, vamos explorar a vetorização de texto usando a biblioteca NLTK para tokenização e a técnica TF-IDF (Term Frequency-Inverse Document Frequency) para representar o texto de forma numérica. Essa vetorização é útil em tarefas de classificação onde se deseja analisar a importância relativa de palavras em documentos e obter uma representação mais adequada para algoritmos de machine learning tradicionais.

- **Tokenização** e **Pré-processamento com NLTK**: Use a NLTK para dividir o texto em tokens (palavras), remover stopwords e normalizar o texto. Isso facilita a análise, pois elimina palavras comuns que não agregam valor semântico e converte o texto em uma forma uniforme.

- **Vetorização** com **TF-IDF**: Aplique a técnica TF-IDF para transformar o texto em uma matriz numérica, onde cada documento é representado por um vetor que indica a importância relativa de cada palavra no corpus.

- **Classificação de Documentos**: Com o texto vetorizado, você pode usar modelos de aprendizado de máquina, como Naive Bayes, SVM, XGBoost e outros para classificar documentos. Considere classificar textos em categorias específicas (como temas ou tópicos) para entender melhor o comportamento da técnica.

- **Exemplo de Código**: 

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Certifique-se de baixar as stopwords do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Função para tokenizar e pré-processar o texto
def preprocess_text(text):
    stop_words = set(stopwords.words("portuguese"))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Exemplo de textos para vetorização
texts = ["Este é um texto de exemplo.", "Outro texto para análise."]
processed_texts = [preprocess_text(text) for text in texts]

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(processed_texts).toarray()

print("Representação TF-IDF:")
print(text_vectors)
```

Crie uma rota `/vectorize` no aplicativo Flask que receba um texto em JSON, processa-o (realizando tokenização e vetorização com TF-IDF) e retorna a representação vetorizada como JSON.

<!--
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Inicialize o aplicativo Flask
app = Flask(__name__)

# Certifique-se de baixar as stopwords do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Configuração do vetor TF-IDF e stopwords
stop_words = set(stopwords.words("portuguese"))
vectorizer = TfidfVectorizer()

# Função de pré-processamento de texto
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Rota para vetorização de texto com TF-IDF
@app.route('/vectorize', methods=['POST'])
def vectorize_text():
    data = request.get_json()
    
    # Extrair o texto
    text_list = data.get('texts', [])
    if not text_list or not isinstance(text_list, list):
        return jsonify({"error": "Lista de textos 'texts' não fornecida ou inválida."}), 400

    # Pré-processar os textos
    processed_texts = [preprocess_text(text) for text in text_list]
    
    # Vetorização com TF-IDF
    text_vectors = vectorizer.fit_transform(processed_texts).toarray()
    
    # Retornar a matriz TF-IDF como JSON
    response = {"tfidf_vectors": text_vectors.tolist()}
    return jsonify(response)

# Executar o aplicativo Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
-->

### Desafio Extra: Utilização de Bases de Dados em NLP

Neste desafio, vocês terão a oportunidade de escolher uma das bases de dados sugeridas para explorar de forma mais rica o potencial do NLP, praticando análise de sentimentos, classificação de texto e vetorização com TF-IDF. Vocês podem escolher uma base para realizar alguma das seguintes tarefas:


- **Análise de Sentimentos**: Usar a pipeline da Hugging Face ou vetorização com TF-IDF e classificadores para prever a polaridade de opiniões.
- **Classificação de Tópicos**: Implementar classificadores usando TF-IDF e embeddings de palavras, categorizando textos em temas específicos.
- **Pré-processamento de Texto**: Aplicar técnicas de NLTK para tokenização, remoção de stopwords e normalização dos textos antes da análise.

Abaixo, encontram-se descrições das bases de dados e links para download: 

- **[IMDB Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**  
   Este dataset de avaliações de filmes da IMDB é amplamente utilizado para **análise de sentimentos**. Com ele, é possível treinar modelos que classificam comentários como positivos ou negativos. A aplicação deste dataset é direta para análise de sentimentos usando a pipeline da Hugging Face ou modelos tradicionais de classificação de texto com TF-IDF.

<!--
- **[Sentiment140](http://help.sentiment140.com/for-students)**  
   O Sentiment140 é uma coleção de tweets rotulados, contendo análises positivas, neutras e negativas. É ideal para projetos de **análise de sentimentos** e **processamento de texto em redes sociais**. O uso desta base desafia vocês a lidar com ruídos e abreviações comuns em tweets, sendo um excelente exercício para pré-processamento de texto com NLTK.
-->

- **[Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)**  
   Com milhões de avaliações de produtos, este dataset é ideal para **classificação de sentimentos** e **identificação de temas**. É possível classificar textos por polaridade ou explorar aspectos de produtos com embeddings de palavras para entender o contexto semântico das opiniões.

- **[Yelp Reviews](https://www.yelp.com/dataset)**  
   O Yelp Open Dataset oferece avaliações de estabelecimentos e pode ser utilizado para análise de sentimentos e classificação por categorias de negócios, como "Restaurantes" ou "Serviços". Esta base é interessante para tarefas de classificação de texto com TF-IDF ou vetorização com embeddings contextuais.

- **[Common Crawl](https://commoncrawl.org/the-data/)**  
   Esta é uma base de dados de texto muito ampla, extraída da web, com diversas possibilidades de análise, desde **classificação de temas** até **modelagem de linguagem**. Ela é ideal para projetos que exigem grandes volumes de dados e permitem aplicar técnicas de embeddings de palavras, como Word2Vec e TF-IDF, para classificação de tópicos ou sumarização.

- **[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)**  
   Esta coleção de posts de newsgroups está organizada em 20 categorias e é excelente para tarefas de **classificação de temas**. Vocês podem treinar modelos para categorizar os textos em tópicos específicos, aplicando vetorização com TF-IDF e classificadores como Naive Bayes ou SVM.

- **[TREC Question Classification](http://cogcomp.org/Data/QA/QC/)**  
   Este dataset de perguntas rotuladas por categoria é excelente para tarefas de **classificação de perguntas** e **resposta a perguntas**. É um ótimo exercício para a criação de pipelines de classificação e para experimentar o uso de embeddings com técnicas de NLTK.

**Boa sorte!**

## 7. Conclusão

O Processamento de Linguagem Natural (NLP) permite que máquinas analisem e respondam ao texto humano em aplicações como atendimento ao cliente, análise de dados e automação de respostas. Técnicas como TF-IDF e modelos avançados (transformers) viabilizam tarefas importantes, como análise de sentimentos, classificação de texto e geração de respostas automáticas. Dominar esses conceitos permite transformar dados textuais em informações úteis, gerando conhecimento e valor em diversas áreas.