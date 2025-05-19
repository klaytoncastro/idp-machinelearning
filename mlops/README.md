# Visão Geral de MLOps: como colocar seu modelo de Machine Learning em produção? 

## 1. Introdução

O Machine Learning (ML) é uma tecnologia cada vez mais presente em aplicações modernas, oferecendo soluções para personalização de conteúdo, automação de decisões e previsões em tempo real. Seu uso vai além da ciência de dados, alcançando áreas como e-commerce, saúde, redes sociais, entre outras. Neste projeto, você criará uma API simples para executar seu modelo de ML, aprenderá sobre roteamento de aplicativos web, conteúdo estático e dinâmico, além de utilizar o depurador para corrigir eventuais erros.

<!--
Nesse cenário, o Flask é um microframework Python para desenvolvimento ágil de aplicativos web, adequado tanto para iniciantes quanto para desenvolvedores mais experientes. Ele é bastante leve e extensível, permitindo expandir facilmente seu aplicativo para operar com bibliotecas mais avançadas, aproveitando todo o poder da linguagem Python e a flexibilidade da web, permitindo que você comece pequeno, escolhendo apenas as peças necessárias, e cresça à medida que seu projeto se desenvolve. 

Também utilizaremos o Docker, uma solução de virtualização leve que permite empacotar aplicações e todas as suas dependências (bibliotecas, configurações e código) em ambientes isolados, chamados containers. Esses containers são altamente portáveis e podem ser executados em qualquer sistema operacional compatível. Essa solução é amplamente adotada no mercado para criar ambientes replicáveis e consistentes, eliminando a necessidade de configurar e instalar manualmente cada aplicação em diferentes máquinas.

---

## 3. Preparação do Ambiente

Nos sistemas Microsoft Windows, recomenda-se a utilização do WSL (Windows Subsystem for Linux) para a instalação do Docker. O WSL é um recurso nativo do Windows que permite a execução de distribuições Linux sem a necessidade de emulação ou virtualização completa, como o Microsoft Hyper-V ou Oracle VirtualBox. Projetado para facilitar o desenvolvimento de software no Windows, o WSL oferece uma integração simplificada entre os dois sistemas operacionais, tornando o uso do Docker mais eficiente e acessível. O uso do Docker, em conjunto com o WSL, é essencial para garantir a replicabilidade do ambiente de desenvolvimento, independentemente do sistema operacional usado por cada estudante.

**Nota**: Usuários de sistemas baseados em Linux ou MacOS não precisam utilizar o WSL, pois esses sistemas já possuem suporte nativo ao Docker. Para executar containers, basta instalar o Docker diretamente, sem a necessidade de qualquer subsistema ou ferramenta adicional.

### Passo 1: Verificação dos Requisitos
Certifique-se de ter uma versão compatível do Windows 10 ou superior e o recurso de virtualização habilitado (VT-x para os processadores da família Intel e AMD-V para os processadores da família AMD). 

### Passo 2: Ativação do WSL
No PowerShell ISE como administrador, execute:

```bash
# Ativa o subsistema Windows para Linux
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Ativa a plataforma de máquina virtual necessária para o WSL 2
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Instala o WSL
wsl --install

# Define a versão 2 do WSL como padrão
wsl --set-default-version 2
```
### Passo 3: Escolha de uma Distribuição

- Caso ainda não utilize uma distribuição Linux embarcada no Windows, instale uma distribuição pela Microsoft Store. Recomenda-se o Ubuntu 24.04 LTS.
- Após a instalação, reinicie o seu computador. 

### Passo 4: Configuração Inicial

- Inicie o aplicativo WSL, configure o usuário e senha da distribuição. Pronto, você já tem acesso a um kernel e terminal Linux. 

### Passo 5: Instalação do Docker

- O Docker Desktop for Windows fornece uma interface gráfica e integra o Docker ao sistema, facilitando a execução e o gerenciamento de containers diretamente no Windows.
- Baixe e instale o [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/#:~:text=Docker%20Desktop%20for%20Windows%20%2D%20x86_64). Após a instalação, o Windows pode solicitar que você faça o logout e o login novamente para aplicar as alterações. 

### Passo 6: Utilização do Ambiente

- Ao longo do curso, você será guiado pelo Professor nas atividades práticas que envolverá o conteúdo das subpastas deste repositório.
- Para começar, inicie o Docker Desktop e, depois, o aplicativo WSL. Se preferir, você pode utilizar o terminal com Powershell e invocar o wsl.exe diretamente no Visual Studio Code (VS Code) para gerenciar seus containers e desenvolver seus projetos.
- A partir daqui, você poderá seguir as instruções do professor para completar os exercícios práticos. Se surgir qualquer dúvida, consulte os materiais de apoio indicados no Canvas e neste repositório. 

-->

## 2. Conceitos de MLOps

Em muitos cenários, construir um modelo de Machine Learning (ML) é apenas o começo. Com o tempo, esses modelos podem se degradar à medida que os dados utilizados para treinamento se tornam obsoletos ou mudam significativamente. Manter o modelo operacional e preciso em produção torna-se um grande desafio.

MLOps (Machine Learning Operations) é uma prática que une o desenvolvimento de modelos de ML com operações de TI (DevOps), garantindo que esses modelos sejam implantados, monitorados, mantidos e escaláveis em ambientes de produção. Seu principal objetivo é garantir que os modelos de ML sejam continuamente integrados, implantados e monitorados com eficiência e confiabilidade.

Para isso, o MLOps abrange todo o ciclo de vida do modelo, desde o treinamento inicial até a implantação e manutenção. Isso inclui práticas como automação e versionamento, que garantem que novos modelos sejam atualizados e testados sem interrupções, evitando falhas e inconsistências. Um aspecto fundamental do MLOps é a Integração Contínua e Implantação Contínua (CI/CD), que permite que novos modelos sejam rapidamente integrados ao ambiente de produção por meio de pipelines automatizados.

O monitoramento contínuo é outra prática essencial do MLOps. Ele inclui o registro de todas as previsões feitas pela API, juntamente com os dados de entrada e o armazenamento dos resultados reais. Com esses dados, é possível comparar as previsões com os resultados reais e calcular métricas de desempenho, avaliando se o modelo está se degradando ao longo do tempo.

Um conceito importante relacionado ao monitoramento é o drift, que ocorre quando os padrões dos dados de entrada ou a relação entre as variáveis e o alvo mudam. O drift de dados reflete mudanças nos padrões dos dados, enquanto o drift de conceito afeta diretamente a capacidade do modelo de realizar previsões corretas. Monitorar a distribuição das variáveis de entrada e o desempenho do modelo ao longo do tempo permite detectar esses problemas.

Finalmente, um pipeline de avaliação contínua é recomendável para garantir que o modelo permaneça confiável. Esse pipeline deve registrar automaticamente as previsões do modelo, armazenar os resultados reais, calcular métricas periodicamente e gerar alertas caso o desempenho caia abaixo de um limite aceitável. Dessa forma, o modelo mantém sua precisão e utilidade em um ambiente de produção.

## 3. Primeiro Passo: Exportação do Modelo Treinado

Uma etapa crucial na implementação de um modelo de ML em produção é a exportação do modelo treinado para um formato que possa ser facilmente carregado e utilizado por aplicações. Geralmente optamos pelo uso do formato `pickle` para realizar essa tarefa. O formato `pickle` oferece uma maneira padrão para serializar objetos em Python. Isso significa que ele pode transformar qualquer objeto Python, incluindo modelos complexos de Machine Learning, em uma sequência de bytes que pode ser salva em um arquivo.

### Por Que Usar o Formato Pickle?

O principal benefício de utilizar o formato `pickle` para exportar modelos de Machine Learning é a sua eficiência e simplicidade em armazenar e recuperar os modelos treinados. Em um cenário de produção, o tempo necessário para treinar um modelo pode ser proibitivo, especialmente com grandes volumes de dados ou algoritmos complexos que requerem alto poder computacional. Assim, treinar o modelo a cada nova requisição de previsão torna-se inviável.

Exportar o modelo treinado como um arquivo `pickle` permite que o modelo seja carregado rapidamente por nossa aplicação Flask, sem a necessidade de reprocessar os dados ou retreinar o modelo. Isso é essencial para garantir a agilidade das respostas em um ambiente de produção, onde a performance e o tempo de resposta são críticos.

### Como Exportar e Carregar um Modelo com Pickle
Exportar um modelo para um arquivo pickle é um processo simples. Primeiro, o modelo é treinado. Após o treinamento, o modelo é serializado com o módulo `pickle` e salvo em um arquivo `.pkl`. O código a seguir exemplifica este processo:

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# Treinando o modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Salvando o modelo em um arquivo pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

```

Para utilizar o modelo em nossa aplicação Flask, simplesmente carregamos o arquivo pickle, deserializamos o objeto e utilizamos para fazer previsões:

```python
# Carregando o modelo do arquivo pickle
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Usando o modelo carregado para fazer previsões
prediction = loaded_model.predict(X_new)
```

## 4. Implantação do Aplicativo Web

### Criando o aplicativo

Clone o repositório `ìdp-machinelearning` e acesse a pasta `production` e o arquivo arquivo `app.py` com o `vim` (use o comando `vim app.py`) ou o editor de sua preferência para dar manutenção ao código, como VS Code.

```bash
cd /opt
git clone https://github.com/klaytoncastro/idp-machinelearning
cd idp-machinelearning/production

```

### Executando o aplicativo 

Vá até o diretório `/opt/idp-machinelearning/production` e suba o container do aplicativo web. 

```bash
docker-compose build
docker-compose up -d
```

Verifique se o contêiner está ativo e sem erros de implantação. 

```bash
docker-compose ps
docker-compose logs
```
Agora, acesse `http://127.0.0.1:5000/example` e verifique o retorno do `.json` de exemplo. 

### Roteamento e visualizações

Roteamento refere-se ao mapeamento de URLs específicas para funções em um aplicativo web. Em outras palavras, quando você acessa um determinado endereço em um navegador web (ou através de uma chamada API), o aplicativo precisa saber qual função deve ser executada e o que deve ser retornado para o usuário. No Flask, isso é feito através do uso de decoradores, como `@app.route()`, para associar funções específicas a URLs. Por exemplo:

```python
@app.route('/inicio')
def inicio():
    return "Página Inicial"
```

Dessa forma, você poderá acessar os *end-points* `http://127.0.0.1:5000/<nome_end-point>` e verá as respectivas páginas em seu navegador. 

### Rotas Dinâmicas

Vamos permitir que os usuários interajam com o aplicativo por meio de rotas dinâmicas. Podemos submeter via método `HTTP POST` um `.json` com as variáveis preditoras e o nosos aplicativo retornará a previsão da variável alvo. Abaixo, exemplo de um vinho de qualidade "ruim": 

```shell
 curl -X POST   -H "Content-Type: application/json"   -d '{
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
      }'   http://localhost:5000/predict
```

- Abaixo, exemplo de código para um vinho de qualidade "boa": 

```shell
 curl -X POST   -H "Content-Type: application/json"   -d '{
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
      }'   http://localhost:5000/predict
```

- Você também pode utilizar um arquivo para fazer `POST` do arquivo `.json`. Seguem exemplos: 

```shell
curl -X POST -H "Content-Type: application/json" -d @bom.json http://localhost:5000/predict
```

```shell
curl -X POST -H "Content-Type: application/json" -d @ruim.json http://localhost:5000/predict
```

- Outra forma de testar a API é com uma extensão como o Postman, diretamente em seu navegador, para fazer as vezes do `curl` mas com uma interface gráfica. 

### Depurando seu aplicativo

O Flask possui um depurador embutido. No nosso ambiente, quando você executa o comando `docker-compose logs`, poderá verificar quais são os eventuais erros e assim corrigir o código de seu aplicativo.

## 5. Tarefa: Coloque outro modelo de ML em Produção

### Objetivo

Nesta atividade, você vai selecionar um problema de classificação ou regressão, treinar um modelo de ML e implementá-lo em produção utilizando Flask como servidor web. O modelo será exportado utilizando a biblioteca `joblib` e o formato `pickle`, permitindo que a API Flask o utilize para fazer previsões a partir de dados recebidos em formato JSON.

### Instruções

Escolha um problema de classificação ou regressão de sua preferência. Por exemplo, você pode optar por utilizar alguns dos datasets que já trabalhamos, como o Air Quality para prever a qualidade do ar, California Housing, para prever o preço de casas, que são tarefas de regressão, ou Bank Marketing para prever se um cliente irá adquirir ou não um produto (classificação) ou, ainda, o pacote `sklearn.datasets`, que disponibiliza alguns conjuntos de dados como o Iris para prever o tipo de uma flor, e outros mais.

### Treinamento do modelo

Utilize o conjunto de dados escolhido para desenvolver e treinar um modelo de ML, optando por um algoritmos como RandomForest, Decision Tree, Linear Regression, ExtraTrees, LightGBM, XGBoost, etc. Após o treinamento, exporte o modelo para um arquivo `.pkl` e adapte a aplicação Flask que apresentamos acima para corresponder à sua escolha.

<!--
Use os arquivos `.ipynb` e `.json` [desta pasta](https://github.com/klaytoncastro/idp-machinelearning/tree/main/production/models) como referência para a exportação do modelo e faça os ajustes necessários.-->

## Conclusão

O MLOps é uma abordagem essencial para garantir que os modelos de Machine Learning sejam confiáveis, escaláveis e facilmente mantidos em produção. A exportação de modelos em formato `pickle` para uso por uma web API com o Flask é uma prática eficiente para a implantação de modelos de ML em produção, disponibilizando rapidamente as capacidades preditivas de modelo um modelo para aplicações em tempo real. No futuro, poderermos explorar outros formatos de exportação de modelo e estratégias de monitoramento contínuo para garantir que o modelo mantenha sua efetividade, diante de mudanças de padrões e comportamentos dos usuários.

<!--

A partir daqui, você pode expandir seu aplicativo, integrando-o com bancos de dados, formulários e aprimorando seu visual com CSS e HTML. 

Você criou um pequeno aplicativo web com o Flask, adicionou rotas estáticas e dinâmicas e aprendeu a usar o depurador. A exportação de modelos em formato pickle é uma prática eficiente para a implantação de modelos de ML em produção, oferecendo uma forma rápida de disponibilizar as capacidades preditivas do modelo com a eficiência necessária para aplicações em tempo real.

O MLOps é uma abordagem essencial para garantir que os modelos de Machine Learning sejam confiáveis, escaláveis e facilmente mantidos em produção. Ele permite que os modelos sejam implantados, monitorados e ajustados de maneira contínua, garantindo que mantenham seu desempenho ao longo do tempo.

Nesta aula, você aprendeu a base para colocar um modelo de Machine Learning em produção. No futuro, poderá explorar outros formatos de exportação de modelo, como joblib, e implementar estratégias de monitoramento contínuo para garantir que o modelo mantenha sua precisão, mesmo diante de mudanças nos dados.

-->