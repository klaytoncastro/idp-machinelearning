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

Nesta tarefa, usaremos a biblioteca Hugging Face Transformers para realizar análise de sentimentos em texto. O modelo pré-treinado de análise de sentimentos permite classificar um texto como "positivo" ou "negativo". Utilizem como base a infraestrutura que implementamos com Flask e implementem uma nova rota na aplicação para avaliar textos utilizando um modelo de classificação de sentimentos pré-treinado da [Hugging Face](https://huggingface.co/blog/sentiment-analysis-python). 

Vocês deverão modificar o código existente para carregar o modelo de análise de sentimentos e criar uma rota `/analyze_sentiment` que aceite requisições `POST` com um `JSON` contendo um texto e retorne o sentimento classificado como 'positivo', 'negativo' ou 'neutro'. Essa rota deve aceitar requisições POST com um JSON contendo o texto a ser analisado, conforme exemplo abaixo: 

```json
{ "text": "Seu texto aqui" }
```

Testem a nova funcionalidade utilizando a ferramenta `curl`, similares aos já mostrados anteriormente na atividade de [MLOps](https://github.com/klaytoncastro/idp-machinelearning/tree/main/mlops): 

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

## 7. Execução com Docker

Esta prática utiliza Docker e Docker Compose para executar a API Flask em container.

Antes de subir a aplicação, crie a rede Docker externa usada no `docker-compose.yml`, caso ela ainda não exista:

```bash
docker network inspect mybridge >/dev/null 2>&1 || docker network create mybridge
docker compose build
docker compose up -d
# Em outra aba de terminal: 
docker compose logs -f
```

- A API ficará disponível em: `http://localhost:5000`
- A documentação no padrão Swagger ficará disponível em: `http://localhost:5000/apidocs`

A aplicação disponibiliza documentação interativa utilizando Flasgger (Swagger UI).

Acesse no navegador: [http://localhost:5000/apidocs](http://localhost:5000/apidocs)

Na interface Swagger:

1. Localize a rota `POST /analyze_sentiment`.
2. Clique em **Try it out**.
3. Informe o JSON:

```json
{
  "text": "This class is excellent and very useful"
}
```

4. Clique em **Execute**.
5. Observe o código HTTP, o sentimento classificado (`POSITIVE` ou `NEGATIVE`) e o score retornado.
6. Repita o procedimento com as rotas `/vectorize`, `/similarity/tfidf`, `/embeddings` e `/similarity/semantic`, utilizando os exemplos das práticas anteriores.
7. Além do terminal usando o `curl`, o Swagger permite pode explorar, testar e validar APIs REST diretamente pelo browser. Além da validação de entradas inválidas (HTTP 400), a aplicação implementa tratamento para rotas inexistentes e métodos HTTP não permitidos.

**Teste 1 — HTTP 404: rota inexistente**

```bash
curl -s -i http://localhost:5000/rota-inexistente
```

Resultado esperado:

```json
{
  "error": "Rota não encontrada"
}
```

**Teste 2 — HTTP 405: método não permitido**

```bash
curl -s -i -X GET http://localhost:5000/vectorize
```

Resultado esperado:

```json
{
  "error": "Método HTTP não permitido"
}
```

### 7.1 Verificar se a API está disponível
```bash
curl http://localhost:5000/
```

### 7.2 Health check

```bash
curl http://localhost:5000/health
```

### 7.3 Análise de sentimentos com Transformers

```bash
curl -X POST http://localhost:5000/analyze_sentiment \
-H "Content-Type: application/json" \
-d '{"text": "This class is excellent and very useful"}'
```

### 7.4 Vetorização com TF-IDF

```bash
curl -X POST http://localhost:5000/vectorize \
-H "Content-Type: application/json" \
-d '{"texts": ["o juiz julgou o processo", "o magistrado analisou os autos", "o processo foi analisado pelo juiz"]}'
```

### 7.5 Sentimento negativo

```bash
curl -s -X POST http://localhost:5000/analyze_sentiment -H 'Content-Type: application/json' -d '{"text":"This application is terrible and completely useless"}' | jq
```

### 7.6 Negação na análise de sentimentos

```bash
curl -s -X POST http://localhost:5000/analyze_sentiment -H 'Content-Type: application/json' -d '{"text":"This movie is not good"}' | jq
```

Compare com o resultado de uma frase afirmativa. Observe como a negação pode alterar a classificação.

### 7.7 Ironia na análise de sentimentos

```bash
curl -s -X POST http://localhost:5000/analyze_sentiment -H 'Content-Type: application/json' -d '{"text":"Great, another terrible experience. Just what I needed!"}' | jq
```

Discuta as limitações do classificador para interpretar ironia e contexto.

### 7.8 Sentimento em português

```bash
curl -s -X POST http://localhost:5000/analyze_sentiment -H 'Content-Type: application/json' -d '{"text":"Esta aula é excelente e muito interessante"}' | jq
```

O modelo utilizado (`distilbert-base-uncased-finetuned-sst-2-english`) foi ajustado para textos em **inglês**. O resultado em português é um experimento, não uma avaliação confiável da polaridade.

### 7.9 Matriz TF-IDF formatada

O endpoint `/vectorize` já retorna `vocabulary`, `processed_texts` e `tfidf_vectors`. Exiba as linhas (documentos) e colunas (termos) em uma tabela:

```bash
curl -s -X POST http://localhost:5000/vectorize -H 'Content-Type: application/json' -d '{"texts":["o juiz julgou o processo","o magistrado analisou os autos","o processo foi analisado pelo juiz"]}' | jq -r '.vocabulary as $v | (["Documento"] + $v | @tsv), (.tfidf_vectors | to_entries[] | (["Doc. \(.key + 1)"] + [.value[] | (. * 10000 | round / 10000 | tostring)] | @tsv))' | column -t -s "$(printf '\t')"
```

Repare que a técnica representa a importância relativa dos termos, não o significado da frase.

### 7.10 Similaridade lexical por cosseno

A similaridade por cosseno **já está implementada** no `app.py`, na rota `/similarity/tfidf`. Não é necessário modificar o código nem reiniciar o serviço para acrescentá-la.

```bash
curl -s -X POST http://localhost:5000/similarity/tfidf -H 'Content-Type: application/json' -d '{"texts":["o juiz julgou o processo","o magistrado analisou os autos","o processo foi analisado pelo juiz"]}' | jq '.similarity_matrix'
```

A matriz compara cada texto com os demais; a diagonal representa a comparação de cada documento consigo mesmo.

### 7.11 Limitação lexical: palavras diferentes, sentidos próximos

```bash
curl -s -X POST http://localhost:5000/similarity/tfidf -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","o magistrado examinou os autos"]}' | jq '{processed_texts, vocabulary, similarity_matrix}'
```

Observe que TF-IDF depende da sobreposição de termos. Embora as frases tenham sentidos próximos, compartilham poucas palavras relevantes.

### 7.12 Negação e similaridade lexical

```bash
curl -s -X POST http://localhost:5000/similarity/tfidf -H 'Content-Type: application/json' -d '{"texts":["This product is good","This product is not good","This product is very good"]}' | jq '{processed_texts, vocabulary, similarity_matrix}'
```

Investigue por que textos com significados opostos podem apresentar alta similaridade lexical.

### 7.13 Embeddings semânticos

A rota `/embeddings` utiliza o modelo multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Cada texto é representado por um vetor de 384 dimensões.

```bash
curl -s -X POST http://localhost:5000/embeddings -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","o magistrado examinou os autos"]}' | jq '{model, dimensions, texts, primeiros_valores: [.embeddings[][:8]]}'
```

Exibimos somente os oito primeiros valores de cada vetor para facilitar a leitura. Na primeira chamada, o carregamento do modelo pode demorar mais.

### 7.14 Similaridade semântica por cosseno

```bash
curl -s -X POST http://localhost:5000/similarity/semantic -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","o magistrado examinou os autos","o restaurante oferece comida italiana"]}' | jq '{model, similarity_matrix}'
```

Diferentemente da similaridade lexical, aqui o cosseno é calculado sobre **embeddings** obtidos a partir dos textos originais.

### 7.15 Comparar TF-IDF e embeddings

Execute as duas rotas com os **mesmos textos** e compare as matrizes:

```bash
curl -s -X POST http://localhost:5000/similarity/tfidf -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","o magistrado examinou os autos","o restaurante oferece comida italiana"]}' | jq '{method, similarity_matrix}'
curl -s -X POST http://localhost:5000/similarity/semantic -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","o magistrado examinou os autos","o restaurante oferece comida italiana"]}' | jq '{method, similarity_matrix}'
```

Identifique quais pares são considerados semelhantes por cada método e explique a diferença.

### 7.16 Similaridade entre idiomas

```bash
curl -s -X POST http://localhost:5000/similarity/semantic -H 'Content-Type: application/json' -d '{"texts":["o juiz analisou o processo","the judge examined the case","o restaurante serve pizza"]}' | jq '{model, similarity_matrix}'
```

Verifique se os embeddings multilíngues aproximam frases equivalentes em português e inglês.

### 7.17 Negação e similaridade semântica

```bash
curl -s -X POST http://localhost:5000/similarity/semantic -H 'Content-Type: application/json' -d '{"texts":["O relatório foi aprovado","O relatório não foi aprovado","O relatório foi rejeitado"]}' | jq '{model, similarity_matrix}'
```

Avalie se o modelo distingue adequadamente afirmação e negação. **Similaridade semântica elevada não garante equivalência lógica.**

### 7.18 Validação de entradas inválidas

A API deve responder com **HTTP 400** quando os dados obrigatórios não forem fornecidos ou estiverem incorretos:

```bash
curl -s -i -X POST http://localhost:5000/analyze_sentiment -H 'Content-Type: application/json' -d '{}'
curl -s -i -X POST http://localhost:5000/vectorize -H 'Content-Type: application/json' -d '{"texts":[]}'
curl -s -i -X POST http://localhost:5000/similarity/tfidf -H 'Content-Type: application/json' -d '{"texts":["apenas um texto"]}'
```

> **Observação:** os testes 7.10 a 7.17 utilizam rotas já existentes no `app.py`. Não é necessário incluir `cosine_similarity` na resposta de `/vectorize`; a similaridade lexical está em `/similarity/tfidf` e a semântica em `/similarity/semantic`. Os comandos com `jq` e `column` pressupõem que essas ferramentas estejam instaladas no terminal.


## 8. Conclusão

Nesta prática, comparamos duas abordagens de NLP:

- TF-IDF: técnica clássica que transforma textos em vetores com base na frequência e importância das palavras.
- Transformers: abordagem moderna que utiliza modelos pré-treinados capazes de capturar contexto linguístico.

Assim, vimos que o processamento de Linguagem Natural (NLP) permite que máquinas analisem e respondam ao texto humano em aplicações como atendimento ao cliente, análise de dados e automação de respostas, análise de sentimentos, classificação de texto, dentre outras aplicações. 

Aprofundar esses conceitos permite transformar dados textuais em informações úteis, gerando conhecimento e valor em diversas áreas.
