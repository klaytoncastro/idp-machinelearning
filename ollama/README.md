# Ollama - Utilizando Modelos Grandes de Linguagem (LLMs)

## 1. Visão Geral

Os **Large Language Models (LLMs)** são modelos de aprendizado profundo treinados para processar e gerar linguagem natural. Eles podem responder a perguntas, resumir documentos, classificar informações, extrair entidades e auxiliar na construção de sistemas inteligentes. Entretanto, **o modelo não consulta automaticamente os documentos de uma organização**: para responder com base em fontes específicas, é preciso disponibilizar esse contexto à aplicação.

O **Ollama** é uma ferramenta que simplifica a **execução e disponibilização de modelos de IA** em computadores e servidores. Ele gerencia o download de modelos, a inferência e a exposição de uma **API HTTP**, permitindo que outras aplicações os utilizem. **Ollama não é um modelo**: é o ambiente que executa modelos como Qwen, Llama, Gemma e Mistral.

Neste laboratório, vamos testar o uso interativo de um LLM para sua integração por API e preparar uma recuperação semântica de documentos.

**Objetivos:**

- Executar o Ollama localmente usando Docker Compose.
- Baixar, listar, executar e inspecionar modelos de linguagem.
- Consumir as APIs `/api/generate`, `/api/chat` e `/api/embed` por `curl` e Python.
- Experimentar *prompt engineering*, temperatura e janela de contexto.
- Diferenciar geração de texto, embeddings, busca semântica e **RAG** (*Retrieval-Augmented Generation*).
- Comparar respostas livres e respostas fundamentadas em fontes, registrando evidências e limitações.

## 2. Conceitos Fundamentais

TF-IDF e embeddings **não são a mesma coisa**. Um modelo de embeddings pode aproximar representações de frases semanticamente relacionadas, mesmo sem coincidência literal de palavras; isso não significa que toda recuperação semântica seja correta. Um **LLM** utiliza parâmetros aprendidos durante o treinamento para estimar sequências de tokens. **Treinamento** é o processo de ajuste desses parâmetros; **inferência** é o uso do modelo já treinado para produzir respostas. O comando `ollama pull` baixa um modelo; `ollama run` realiza inferência — **nenhum deles treina o modelo**.

| Abordagem | O que produz | Uso típico |
|---|---|---|
| **TF-IDF** | Vetores de importância estatística de termos | Busca lexical e classificação tradicional |
| **Embeddings** | Vetores densos que representam aspectos semânticos | Busca por similaridade, agrupamento, RAG |
| **LLM** | Texto gerado em resposta a um prompt | Explicações, sínteses, perguntas e respostas |
| **RAG** | Resposta do LLM enriquecida com documentos recuperados | Perguntas fundamentadas em bases de conhecimento |


### 2.1. Modelo de linguagem, parâmetros e inferência

- **Tokens:** unidades de texto processadas pelo modelo (não são necessariamente palavras completas).
- **Parâmetros do modelo:** valores aprendidos durante o treinamento; modelos maiores costumam exigir mais recursos.
- **Quantização (ex.: Q4):** representação de pesos com menor precisão para reduzir o consumo de memória, com possíveis efeitos na qualidade.
- **Janela de contexto (`num_ctx`):** quantidade de tokens que o modelo pode considerar durante uma solicitação. Aumentá-la costuma exigir mais memória.
- **Temperatura (`temperature`):** controla a aleatoriedade da geração. Valores baixos tendem a produzir respostas mais estáveis, mas **não garantem correção factual**.
- **Alucinação:** resposta plausível, porém sem sustentação nas fontes ou incorreta.

### 2.2. Ollama, modelo, aplicação e dados

```text
Usuário / Aplicação (Python, Flask, Jupyter, Streamlit)
                        |
                        | HTTP / JSON
                        v
              Ollama (porta 11434)
                        |
                        v
           Modelo local de linguagem
             (Qwen, Llama, Gemma...)

           Documentos e bases externas
                    |
            Recuperação / RAG
                    |
           Contexto enviado ao LLM
```

O Ollama é responsável por **servir o modelo**; a aplicação é responsável por autenticação, regras, histórico, obtenção de documentos, recuperação, controle de acesso e avaliação da resposta. 

---

## 3. Preparação do Ambiente

### 3.1. Pré-requisitos

- Docker e Docker Compose (`docker compose`) funcionando.
- `curl` para testar a API; Python 3 para a etapa de integração.
- Espaço disponível para baixar os modelos; internet durante o primeiro download.
- Memória suficiente para executar o modelo selecionado. O uso de **GPU é opcional** e pode acelerar significativamente a inferência; em CPU, as respostas poderão demorar mais.

**Modelos sugeridos:**

| Modelo | Finalidade | Download aproximado |
|---|---|---:|
| `qwen3:4b-instruct-2507-q4_K_M` | Geração de texto em português | 2,5 GB |
| `qwen3-embedding:0.6b` | Embeddings multilíngues | 639 MB |
| `qwen3:1.7b` | Comparação com um modelo menor | 1,4 GB |

> Os tamanhos referem-se aos arquivos divulgados na biblioteca Ollama, **não** à memória total consumida em execução. Modelos maiores, contexto mais longo e requisições simultâneas demandam mais recursos. Use o modelo opcional apenas se houver tempo e capacidade de hardware.

### 3.2. Estrutura do laboratório

```text
ollama/
├── README.md
├── docker-compose.yml          # base: CPU, funciona em qualquer sistema
├── docker-compose.nvidia.yml   # overlay opcional: GPU NVIDIA
├── requirements.txt
└── rag_minimo.py
```

O arquivo `docker-compose.yml` contém a configuração base, que executa em **CPU** e funciona em Linux, Windows e macOS:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-lab
    restart: unless-stopped
    ports:
      - "127.0.0.1:11434:11434"
    environment:
      OLLAMA_NO_CLOUD: "1"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

O volume mantém os modelos baixados mesmo que o contêiner seja recriado. A porta publicada em `127.0.0.1` permite acesso **apenas pelo próprio computador**. A variável `OLLAMA_NO_CLOUD=1` desativa recursos de nuvem do Ollama, mantendo o laboratório no modo local.

### 3.2.1. GPU NVIDIA (opcional)

- **Linux:** o NVIDIA Container Toolkit (Ubuntu/Debian):

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

### 3.3. Iniciar e verificar o serviço

Execute no diretório `ollama/`:

```bash
# CPU (qualquer sistema)
docker compose up -d

# GPU NVIDIA (Linux / Windows)
# docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d

docker compose ps
curl http://localhost:11434/api/tags

# Confirmar CPU ou GPU: library=cpu ou library=CUDA
docker compose logs ollama | grep "inference compute"
```

A rota `/api/tags` retorna os modelos locais disponíveis; na primeira execução, `models` poderá estar vazio. Após carregar um modelo, `docker compose exec ollama ollama ps` mostra `100% GPU` ou `100% CPU` na coluna `PROCESSOR`; uma divisão como `40%/60% CPU/GPU` indica que o modelo não coube inteiro na VRAM.

### 3.4. Baixar os modelos

```bash
docker compose exec ollama ollama pull qwen3:4b-instruct-2507-q4_K_M
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama list
```

**Opcional, para comparar modelos de diferentes tamanhos:**

```bash
docker compose exec ollama ollama pull qwen3:1.7b
```

---

## 4. Formas de interação com o Modelo

### 4.1. Conversa pelo terminal

```bash
docker compose exec ollama ollama run qwen3:4b-instruct-2507-q4_K_M
```

Digite a pergunta abaixo no chat interativo:

> Explique a diferença entre Machine Learning, Deep Learning e modelos de linguagem, usando um exemplo de cada.

Digite `/bye` para sair. Experimente também:

```bash
docker compose exec ollama ollama show qwen3:4b-instruct-2507-q4_K_M
docker compose exec ollama ollama ps
```

**Observe:** o Ollama fornece a interface e o gerenciamento de execução, mas o conteúdo da resposta é produzido pelo **modelo**.

### 4.2. Geração por API: `/api/generate`

Execute em um terminal da sua máquina (não dentro do contêiner):

```bash
curl -s http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "prompt": "Explique em português o que é uma JVM e para que serve o garbage collector.",
    "stream": false,
    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 250}
  }' | python3 -m json.tool
```

**Por que `stream: false`?** Por padrão, a API pode transmitir várias partes da resposta em sequência. Com `false`, o resultado é um único objeto JSON, mais fácil de examinar e comparar. No retorno, identifique `response`, `prompt_eval_count`, `eval_count` e `total_duration` (duração em nanossegundos).

### 4.3. Conversa com papéis: `/api/chat`

```bash
curl -s http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "stream": false,
    "messages": [
      {"role": "system", "content": "Você é um monitor de Machine Learning. Responda em português, com precisão e de forma sucinta."},
      {"role": "user", "content": "Qual a diferença entre TF-IDF e embeddings semânticos?"}
    ],
    "options": {"temperature": 0, "num_predict": 250}
  }' | python3 -m json.tool
```

Na API de chat, a resposta estará em `message.content`. Para manter uma conversa, a **aplicação deve reenviar o histórico de mensagens** relevante em novas requisições; a chamada REST isolada não mantém um histórico de conversa automaticamente.

---

## 5. Atividades

### Tarefa 1: Testar a influência dos parâmetros de geração

Faça a mesma pergunta três vezes, mudando **somente** `temperature` entre `0`, `0.5` e `1.0`, mantendo o mesmo modelo, prompt e `num_ctx`.

Pergunta sugerida:

> Proponha três hipóteses para o aumento do tempo de resposta de uma aplicação Java, distinguindo hipótese de diagnóstico confirmado.

Registre as respostas e compare a diversidade, a consistência e a tendência de apresentar afirmações sem evidência. Para experimentos mais controlados, mantenha também `seed` e `num_predict` fixos.

**Questão:** temperatura baixa elimina erros factuais? Justifique usando suas próprias observações.

### Tarefa 2: Respostas sem contexto versus respostas fundamentadas

**Etapa A — sem fornecer a fonte:**

```bash
curl -s http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Qual o valor de -Xmx configurado no servidor lab-app-01?"}
    ],
    "options": {"temperature": 0}
  }' | python3 -m json.tool
```

Como o servidor `lab-app-01` é **fictício**, o modelo não recebeu informações suficientes para conhecer sua configuração.

**Etapa B — fornecendo um documento fictício ao modelo:**

```bash
curl -s http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "stream": false,
    "messages": [
      {"role": "system", "content": "Responda somente com base na fonte apresentada. Cite o ID da fonte. Se não houver informação suficiente, informe a limitação."},
      {"role": "user", "content": "FONTE DOC-02: A instância lab-app-01 utiliza a JVM com -Xmx4g e G1GC. PERGUNTA: Qual o valor de -Xmx configurado no servidor lab-app-01?"}
    ],
    "options": {"temperature": 0}
  }' | python3 -m json.tool
```

**Questões:**

1. Qual resposta possui evidência verificável? A referência `DOC-02` apareceu corretamente?
2. Se a pergunta solicitar a **causa definitiva de uma lentidão**, apenas a configuração de `-Xmx` permite concluir isso?
3. Fornecer contexto no prompt é uma forma simples de *grounding*. O que ainda falta para construir um **RAG real**, com seleção automática de fontes?

### Tarefa 3: Gerar embeddings com `/api/embed`

Use um modelo específico de **embeddings**, diferente do modelo de geração de texto:

```bash
curl -s http://localhost:11434/api/embed \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embedding:0.6b",
    "input": [
      "O servidor apresentou pausas prolongadas no coletor de lixo da JVM.",
      "A aplicação teve lentidão devido às pausas de garbage collection.",
      "O tribunal publicou um novo calendário de feriados."
    ]
  }' > embeddings.json
```

O resultado terá a propriedade `embeddings`, que contém **um vetor para cada texto**. Confira a quantidade de vetores e a dimensão de cada um:

```bash
python3 -c 'import json; v=json.load(open("embeddings.json"))["embeddings"]; print("Quantidade:", len(v), "| Dimensões:", [len(x) for x in v])'
```

**Questão:** os dois primeiros textos compartilham significado, mesmo usando palavras diferentes? Por que isso importa para recuperar documentos em uma base?

### Tarefa 4: Recuperação semântica e RAG mínimo em Python

O arquivo [`rag_minimo.py`](./rag_minimo.py) contém um exemplo executável, ainda sem banco vetorial externo. Ele realiza:

1. Preparação de um pequeno conjunto de documentos fictícios, identificados como `DOC-01` a `DOC-04`.
2. Geração dos embeddings de todos os documentos e da pergunta.
3. Cálculo da **similaridade por cosseno** e seleção dos três documentos mais próximos.
4. Envio dos documentos selecionados, com seus identificadores, ao modelo de linguagem.
5. Apresentação das fontes recuperadas, resposta e métricas de inferência.


<!--
Instale a dependência em um ambiente virtual Python e execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rag_minimo.py
```
-->

**Observe:** o ranking por similaridade não verifica se a fonte é verdadeira. O LLM também pode interpretar mal um documento ou citar uma fonte inadequada. A aplicação deve avaliar a recuperação **e** a resposta.

### Tarefa 5: Comparar modelos (opcional)

Faça o download de `qwen3:1.7b` e repita as perguntas das tarefas 1 e 2, substituindo apenas o nome do modelo. Compare **qualidade, tempo de execução e recursos consumidos**, sem pressupor que o modelo menor necessariamente produzirá respostas piores em todas as questões.

```bash
docker compose exec ollama ollama pull qwen3:1.7b
docker compose exec ollama ollama ps
```

---

## 6. Interpretação dos Resultados

### 6.1. Métricas disponíveis na API

| Campo | Significado |
|---|---|
| `total_duration` | Tempo total da operação, em nanossegundos |
| `load_duration` | Tempo para carregar o modelo, quando aplicável |
| `prompt_eval_count` | Quantidade de tokens processados no prompt |
| `eval_count` | Quantidade de tokens produzidos pelo modelo |
| `eval_duration` | Tempo gasto na geração dos tokens, em nanossegundos |

É possível estimar a taxa de geração em **tokens/s** como `eval_count / (eval_duration / 1e9)`, quando `eval_duration` for diferente de zero. Compare execuções com o **mesmo prompt, ambiente, modelo e parâmetros**, considerando o impacto do carregamento inicial e da diferença entre CPU e GPU.

### 6.2. Grounding, citação e qualidade da recuperação

- **Grounding:** a resposta é sustentada pelo conteúdo efetivamente fornecido?
- **Referência:** os IDs citados correspondem a documentos recuperados e realmente sustentam as afirmações?
- **Recuperação:** o documento necessário foi selecionado pelo sistema?
- **Alucinação:** o modelo apresentou como fato algo ausente das fontes?
- **Abstenção:** o modelo reconheceu corretamente quando faltava informação?

**Uma resposta fluente não é, necessariamente, uma resposta correta.** Métricas mais elaboradas, como **Reference F1** e avaliação por LLM, dependem de um conjunto de perguntas com respostas e referências de comparação. 

---

## 7. Diagnóstico e Encerramento

```bash
# Inspecionar o estado do serviço
docker compose ps
curl http://localhost:11434/api/tags

# Conferir os modelos instalados e os atualmente carregados
docker compose exec ollama ollama list
docker compose exec ollama ollama ps

# Visualizar os últimos logs
docker compose logs --tail=100 ollama

# Parar os serviços sem apagar os modelos
docker compose down
```

**Problemas comuns:**

- **`Connection refused`:** verifique se o contêiner está em execução e se a porta 11434 não está ocupada por outro processo (inclusive uma instalação nativa do Ollama).
- **`model not found`:** confirme o nome/tag e execute o `ollama pull` correspondente.
- **Resposta muito lenta:** verifique se o modelo está usando CPU ou GPU (`ollama ps`), reduza o tamanho do modelo ou a janela de contexto se necessário.
- **Memória insuficiente:** experimente o modelo opcional `qwen3:1.7b` em vez do modelo 4B; a disponibilidade depende do hardware.
- **`rag_minimo.py` falhou:** confirme que os dois modelos foram baixados e que `requests` foi instalado no mesmo ambiente Python utilizado.

> **Atenção:** `docker compose down -v` remove também o volume persistente deste projeto e, portanto, os modelos baixados. Não use `-v` se quiser preservá-los.

## 8. Conclusão

Neste laboratório, evoluímos do consumo de NLP tradicional para a **inferência com LLMs locais**, passando por execução em contêiner, API REST, controle de parâmetros, embeddings e uma implementação mínima de recuperação semântica com geração aumentada por contexto.

O principal aprendizado é a separação de responsabilidades: **o Ollama executa modelos; a aplicação recupera, seleciona, valida e fornece os dados que fundamentarão a resposta**. Para transformar uma demonstração em um sistema confiável, é necessário incorporar qualidade das fontes, controle de acesso, avaliação, rastreabilidade e observabilidade.

### Referências

- [Ollama — documentação oficial](https://docs.ollama.com/)
- [Ollama — execução com Docker](https://docs.ollama.com/docker)
- [Ollama — API de chat](https://docs.ollama.com/api/chat)
- [Ollama — API de embeddings](https://docs.ollama.com/api/embed)
- [Ollama — parâmetros e Modelfile](https://docs.ollama.com/modelfile)
- [Biblioteca de modelos Qwen3](https://ollama.com/library/qwen3/tags)
- [Biblioteca Qwen3 Embedding](https://ollama.com/library/qwen3-embedding/tags)


<!--
## Entrega da Atividade

Produza um **relatório curto** (PDF ou Markdown) com seu nome, identificação dos experimentos e evidências reais da execução:

1. **Ambiente:** captura de `docker compose ps` e `ollama list`, mostrando os modelos disponíveis.
2. **Geração:** comandos e resultados de ao menos dois testes de temperatura com o mesmo prompt.
3. **Grounding:** respostas da Tarefa 2, com e sem a fonte `DOC-02`, e análise crítica.
4. **Embeddings e RAG:** número de vetores/dimensões, ranking recuperado em `rag_minimo.py`, resposta gerada e referências citadas.
5. **Conclusão:** em até dez linhas, explique o que o Ollama resolve, o que é responsabilidade da aplicação e por que um RAG não elimina alucinações automaticamente.

**Critério de avaliação:** execução demonstrada, interpretação técnica dos resultados, rastreabilidade das evidências e capacidade de identificar limitações. **Não basta apresentar comandos copiados sem saída ou análise.** Não utilize senhas, tokens, documentos sigilosos ou dados de produção nos testes e nas capturas.
-->