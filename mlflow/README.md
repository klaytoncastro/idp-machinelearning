# Laboratório MLOps: Git, DVC e MLflow

O objetivo desta prática é versionar um dataset CSV e rodar um pipeline com DVC, registrar execuções no MLflow e comparar resultados. O laboratório usa `data/avaliacoes.csv`, um conjunto sintético para fins didáticos. O modelo estima `final_grade` a partir das variáveis de estudo e frequência.

- **Git** registra código, configurações, ponteiros do DVC e o estado do pipeline.
- **DVC** acompanha o conteúdo do CSV, seus hashes, parâmetros e métricas do pipeline.
- **MLflow Tracking** registra execuções, parâmetros, métricas, identificação do conjunto de dados e artefato do modelo.
- O contêiner de treinamento permanece ativo para uso em terminal interativo.


## Estrutura

```text
mlflow/
├── data/avaliacoes.csv
├── src/
│   ├── train.py
│   └── change_data.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

Depois de executar `dvc init` e `dvc add`, o DVC também cria arquivos como `.dvc/`, `.dvcignore` e `data/avaliacoes.csv.dvc`.

## Subir os contêineres

Na pasta `mlflow`, execute:

```bash
docker compose up -d --build
```

O DVC está em `requirements.txt`; esse build instala a dependência na imagem. Se você alterar `requirements.txt`, reconstrua a imagem com o mesmo comando.

Abra o MLflow em [http://localhost:5001](http://localhost:5001). Entre no contêiner de treinamento:

```bash
docker exec -it mlops-lab-trainer bash
```

Os comandos seguintes são executados dentro do contêiner, em `/workspace`.

## 1. Inicializar o DVC e versionar o CSV

O diretório `/workspace` já está dentro de um repositório Git. Confirme a raiz:

```bash
git rev-parse --show-toplevel
git status
```

Não execute `git init` se o repositório já existir. No ambiente da prática, o Git já estava inicializado; por isso, `git init -b main` apenas o reinicializou e manteve a branch `master`.

Inicialize o DVC uma única vez:

```bash
dvc init
```

Como o CSV já havia sido adicionado ao Git nesse ambiente, remova-o do índice do Git sem apagar o arquivo local. Depois, adicione-o ao DVC:

```bash
git rm --cached -- data/avaliacoes.csv
dvc add data/avaliacoes.csv
git status
git add .
git commit -m "Versiona o CSV com DVC"
```

A mensagem `rm 'data/avaliacoes.csv'` informa que o Git deixou de acompanhar o arquivo diretamente. O CSV continua na pasta de trabalho. O Git passa a registrar `data/avaliacoes.csv.dvc`, que contém o identificador da versão; o conteúdo fica no cache local do DVC.

Se o CSV ainda não estiver rastreado pelo Git, pule o comando `git rm --cached`.

## 2. Executar o pipeline

```bash
dvc repro
dvc metrics show
git status
git add dvc.lock metrics.json
git commit -m "Registra execução de referência"
```

O DVC lê os parâmetros de `params.yaml`, executa `src/train.py` e atualiza `metrics.json`. O treinamento também envia parâmetros, métricas e o modelo ao experimento `avaliacoes-academicas` no MLflow.

O `train.py` precisa passar `skops_trusted_types=["sklearn.tree._tree.Tree"]` para `mlflow.sklearn.log_model`. Essa classe é usada internamente pelo Random Forest e, sem essa permissão explícita, o MLflow interrompe o registro do modelo. 

## 3. Comparar duas execuções — 22 a 32 min

Altere `n_estimators` de `30` para `100` em `params.yaml` e rode:

```bash
dvc repro
dvc metrics diff HEAD
```

No MLflow, compare as execuções. Observe os parâmetros, `mae`, `rmse`, `r2`, `dataset_sha256` e o artefato do modelo. O hash do CSV deve permanecer igual; nesta comparação, o parâmetro alterado foi o número de árvores.

Exemplo: 

| Métrica | 30 árvores | 100 árvores |
|---|---:|---:|
| MAE | 4,1416 | 4,1112 |
| RMSE | 5,3607 | 5,3033 |
| R² | 0,8695 | 0,8723 |

A melhora é pequena e vale para essa divisão dos dados. Uma execução não demonstra que 100 árvores sejam sempre melhores.

Registre a segunda execução:

```bash
git add params.yaml dvc.lock metrics.json
git commit -m "Compara modelo com 100 árvores"
```

## 4. Versionar uma nova versão do CSV

O script `src/change_data.py` altera os dados para demonstrar uma nova versão: ele soma duas horas de estudo às dez primeiras linhas, respeitando o limite definido no código. Isso é uma alteração controlada. Como o arquivo pode estar protegido pelo DVC, libere a cópia de trabalho antes de executar o script:

```bash
dvc unprotect data/avaliacoes.csv
python src/change_data.py
dvc add data/avaliacoes.csv
git diff -- data/avaliacoes.csv.dvc
dvc repro
dvc metrics show
```

No MLflow, verifique se o `dataset_sha256` mudou. O DVC registra o novo ponteiro do CSV; o MLflow registra a execução produzida com essa versão. Salve o ponteiro e o resultado no Git:

```bash
git add src/change_data.py data/avaliacoes.csv.dvc dvc.lock metrics.json
git commit -m "Registra nova versão do CSV"
```

Para restaurar a versão anterior, localize o commit anterior à alteração e recupere o ponteiro:

```bash
git log --oneline -- data/avaliacoes.csv.dvc
git checkout HASH_ANTERIOR -- data/avaliacoes.csv.dvc
dvc checkout
dvc repro
```

O Git recupera o ponteiro versionado; o `dvc checkout` materializa o CSV correspondente a partir do cache local. O `dvc repro` atualiza o treinamento para os dados restaurados.

## Parar os serviços

```bash
docker compose down
```

O banco e os artefatos do MLflow permanecem no volume nomeado. Para apagá-los também:

```bash
docker compose down -v
```

## Considerações Finais

DVC e MLflow organizam versões e evidências da execução; não garantem, por si só, que o desenho da avaliação esteja correto. A comparação desta prática mantém os dados e a partição fixos e altera um parâmetro do modelo.

<!--
Providenciar MinIO/S3 para um remoto DVC e GitLab local para hospedagem e CI/CD. 
1. O que o Git guarda sobre o CSV e onde fica seu conteúdo?
2. Qual é a diferença entre o papel do DVC e o do MLflow?
3. Como identificar quais dados e parâmetros produziram um modelo?
4. O que mudou entre as execuções de 30 e 100 árvores?
-->