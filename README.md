# Introdução ao Aprendizado de Máquina

Bem-vindo ao repositório do curso de **Aprendizado de Máquina**! Este espaço foi criado para proporcionar uma experiência prática e imersiva em Machine Learning, por meio de desafios que desenvolvem habilidades em análise de dados, modelagem preditiva, visualização e storytelling.

## 1. Visão Geral

O **Aprendizado de Máquina (Machine Learning - ML)** é um ramo da **Inteligência Artificial** que permite que computadores aprendam **padrões e comportamentos** a partir de dados, sem a necessidade de programação explícita para cada tarefa.  

### Definição Clássica  

> *"Diz-se que um programa de computador aprende com a experiência **E**, com relação a alguma tarefa **T** e alguma medida de desempenho **P**, se seu desempenho em **T**, medido por **P**, melhora com a experiência **E**."* (Mitchell, 1997)  

Em outras palavras, um sistema aprende quando, ao ser exposto a mais dados (**experiência E**), ele melhora seu desempenho (**P**) em determinada tarefa (**T**). Ou seja, na **programação tradicional** utilizamos linguagens como **C ou Java** para escrever um conjunto de **regras de negócio** que processam **dados de entrada** e produzem uma **saída**:

```text
Entrada (Dados) + Regras (Código) → Saída
```

Já com Machine Learning, em vez de definir essas regras manualmente, a abordagem do programa é aprender essas regras automaticamente a partir dos dados:

```text
Entrada (Dados) + Saída (Rótulos) → Algoritmo aprende as Regras
```

Por exemplo, se quisermos escrever um programa em abordagem tradicional para classificar e-mails como spam ou não spam, precisaríamos definir regras fixas, como:

- Se o e-mail contém palavras como "dinheiro fácil", "ganhe agora", então ele é SPAM.
- Se o remetente está na lista de contatos, então NÃO é SPAM.

Esse tipo de abordagem depende totalmente das regras que foram programadas, o que pode ser limitado e difícil de manter conforme novas variações aparecem. Já em Machine Learning, temos o contrário, o modelo se adapta a novos padrões e variações sem precisar reescrever as regras manualmente: 

- O algoritmo analisa milhares de e-mails rotulados como spam ou não spam.
- O modelo descobre padrões nesses e-mails, identificando quais palavras, remetentes e estruturas são mais comuns em cada categoria.
- Depois, o modelo pode classificar novos e-mails automaticamente, mesmo sem ter regras programadas manualmente.

---

### Principais Tipos de Aprendizado  

O aprendizado de máquina pode ser dividido em **quatro categorias principais**, dependendo da forma como os dados são apresentados ao modelo e do objetivo do aprendizado.  

| **Tipo de Aprendizado**        | **Descrição** | **Exemplo de Aplicação** |
|--------------------------------|--------------|--------------------------|
| **Aprendizado Supervisionado**  | O modelo aprende a partir de um conjunto de dados **rotulado** (com respostas conhecidas). | Classificação de e-mails como spam ou não-spam. |
| **Aprendizado Não Supervisionado** | O modelo encontra **padrões ocultos** nos dados sem que haja respostas pré-definidas. | Agrupamento de clientes por comportamento de compra (*clustering*). |
| **Aprendizado Semi-Supervisionado** | Combina aprendizado supervisionado e não supervisionado, utilizando um **pequeno conjunto de dados rotulado** e um grande conjunto de dados sem rótulo. | Diagnóstico médico onde apenas alguns exames têm diagnóstico confirmado. |
| **Aprendizado por Reforço** | O modelo interage com um **ambiente dinâmico**, recebendo **recompensas ou penalidades** por suas ações. | Treinamento de robôs e agentes para jogar xadrez ou dirigir veículos autônomos. |

---

### Por que Machine Learning é importante?  

O aprendizado de máquina está presente em diversas **aplicações do nosso dia a dia**, como:  

- **Recomendações personalizadas** – Filmes e músicas (*YouTube*, *Netflix, Spotify*).  
- **Assistentes Virtuais** – Alexa, Google Assistant, Siri.  
- **Diagnósticos médicos automatizados** – Modelos que auxiliam médicos na detecção de doenças.  
- **Carros autônomos** – Sistemas que aprendem a dirigir de forma segura e eficiente.  

Neste curso, você terá a oportunidade de **explorar diferentes abordagens de Machine Learning**, aplicando-as a desafios reais e desenvolvendo modelos capazes de tomar decisões **baseadas em dados**. 

## 2. Como Usar Este Repositório

O repositório é organizado em subpastas, cada uma correspondente a um desafio específico. Dentro de cada pasta, você encontrará um `README.md` com detalhes do desafio, incluindo objetivos de aprendizado, tarefas a serem realizadas e critérios de avaliação.

- **Instruções:** Cada desafio vem com um conjunto de orientações e requisitos necessários para completá-lo com sucesso.

- **Recursos Adicionais:** Para ajudá-lo a entender melhor os conceitos ou ferramentas específicas usadas nos desafios consulte o material disponível no Canvas. 

- **Feedback e Avaliação:** Ao longo do curso, você terá a oportunidade de receber feedback sobre seu trabalho e ver como seus colegas abordaram os mesmos problemas.

## 3. Desafios de Machine Learning

Cada desafio é uma experiência prática, onde você aplicará o conhecimento adquirido para explorar novas ferramentas, técnicas e resolver problemas reais. Desde a análise exploratória de dados até a construção e otimização de modelos de Machine Learning, você terá a oportunidade de aprofundar seu aprendizado de forma progressiva.

<!--

| #  |Tarefa                          | Tipo de Problema | Prazo      |
|----|--------------------------------|------------------|------------|
| 01 | [Wine Quality](./winequality/) | Classificação    | 14/03/2025 |
| 02 | [Air Quality](./airquality/)   | Regressão        | 28/03/2025 |
| 03 | [Bank](./bank/)                | Agrupamento      | 18/04/2025 |

| 02 | [Wine Quality - Tarefa de Regressão](./airquality/) | 28/02/2025 |

Submeta aqui todos os notebooks em formato .ipynb contendo a resolução das atividades que você realizou individualmente durante nossas aulas práticas. 
•	Air Quality Dataset:
https://github.com/klaytoncastro/idp-machinelearning/tree/main/airqualityLinks to an external site.
•	Bank Dataset:
https://github.com/klaytoncastro/idp-machinelearning/tree/main/bankLinks to an external site.
•	California Dataset:
https://github.com/klaytoncastro/idp-machinelearning/tree/main/californiaLinks to an external site.
•	Mall Customers Dataset:
https://github.com/klaytoncastro/idp-machinelearning/tree/main/clusteringLinks to an external site.
•	Iris Dataset:
https://github.com/klaytoncastro/idp-machinelearning/tree/main/irisLinks to an external site.
•	Sentiment Analysis
https://github.com/klaytoncastro/idp-machinelearning/tree/main/nlpLinks to an external site.
Instruções:
1.	Nomeie seu arquivo como nome_atividade_nome_aluno.ipynb.
Exemplo: airquality_joao_silva.ipynb.
2.	Certifique-se de que o notebook está funcionando corretamente antes de enviar.
3.	Submeta o arquivo até a data limite indicada.
O representante de cada grupo indicado nas atividades abaixo deve submeter o notebook .ipynb contendo a solução desenvolvida durante as aulas práticas. Certifique-se de que está funcionando corretamente antes de enviar e, registre os nomes dos integrantes do grupo na primeira célula do notebook.
Atividades:
•	Reinforcement LearningLinks to an external site.
•	RNA (Redes Neurais Artificiais)Links to an external site.
•	Rules (Regras de Associação)Links to an external site.
Instruções: Nomeie o arquivo como grupoX_tarefa.ipynb, onde X é o número do grupo (exemplo: grupo1_reinforcement.ipynb).
-->

Fique atento(a) às atualizações, pois novos desafios serão adicionados ao longo do curso!

### Colabore e Aprenda Junto!

Seu engajamento e troca de conhecimento são fundamentais para o sucesso do curso. Participe das discussões, compartilhe insights e aprenda com diferentes abordagens!

- **Boa Sorte!**