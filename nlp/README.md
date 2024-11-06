# NLP (Processamento de Linguagem Natural)

NLP, ou Processamento de Linguagem Natural, é uma área da Inteligência Artificial que abrange todas as tarefas relacionadas ao entendimento e à geração de texto em linguagem humana. O NLP permite que máquinas interpretem, analisem e respondam a textos de forma inteligente, facilitando a interação entre pessoas e sistemas. Entre as tarefas comuns de NLP, encontramos análise de sentimentos, classificação de texto, resposta a perguntas e tradução automática.

Os LLMs (Large Language Models), como GPT-4, LLaMA e Claude, são modelos de NLP em grande escala, com bilhões de parâmetros. Esses modelos avançados têm uma capacidade profunda de compreender e gerar texto, abrindo novas possibilidades para aplicações complexas em linguagem.

## Principais Tarefas em NLP

Aqui estão algumas tarefas importantes e amplamente aplicáveis no campo de NLP:

### 1. Análise de Sentimentos
A análise de sentimentos é uma tarefa de NLP que identifica a emoção ou polaridade de um texto, classificando-o como positivo, negativo ou neutro. É usada para entender a percepção dos usuários em redes sociais, avaliações de produtos e feedbacks de clientes, entre outros.

### 2. Classificação de Texto
A classificação de texto envolve categorizar frases, parágrafos ou documentos inteiros em categorias específicas. Por exemplo, classificar e-mails como “Spam” ou “Não Spam” ou agrupar notícias em categorias como “Esportes”, “Política” e “Tecnologia”.

### 3. Resposta a Perguntas
Nesta tarefa, o modelo recebe uma pergunta e um contexto, e deve responder com base nas informações fornecidas. Isso é útil em assistentes virtuais e chatbots que precisam responder a consultas específicas dos usuários com precisão.

### 4. Geração de Texto
A geração de texto permite que um modelo complete frases, redija parágrafos ou crie conteúdo original a partir de uma entrada inicial. Usada para gerar artigos, auxiliar em escrita criativa ou responder a mensagens de maneira personalizada.

### 5. Reconhecimento de Entidades Nomeadas (NER)
O reconhecimento de entidades nomeadas (NER) identifica e classifica entidades específicas em um texto, como nomes de pessoas, locais, organizações, datas e quantidades. Isso é valioso em análise de documentos, mineração de dados e na extração de informações estruturadas de textos não estruturados.

## Tarefa

Usem como base a infraestrutura que implementamos com Flask em [Production](). Implementem uma nova rota em nossa aplicação Flask para avaliar textos utilizando um modelo de classificação de sentimentos pré-treinado da [Hugging Face](https://huggingface.co/blog/sentiment-analysis-python). 

Vocês deverão modificar o código existente para carregar o modelo de sentimentos e criar uma rota `/analyze_sentiment` que aceite requisições `POST` com um `JSON` contendo um texto e retorne o sentimento classificado como 'positivo', 'negativo' ou 'neutro'. Utilizem o seguinte formato `JSON` para a requisição: 

```json
{ "text": "Seu texto aqui" }
```
Testem a nova funcionalidade utilizando comandos `curl` similares aos mostrados anteriormente. Abaixo, segue código exemplo para importação de uma biblioteca popular de análise de sentimentos:

```python
# Importar Biblioteca da Hugging Face
from transformers import pipeline

# Carregar o modelo de classificação de sentimentos
sentiment_model = pipeline("sentiment-analysis")

```
