# Laboratório MLOps: Git, DVC e MLflow

O objetivo desta prática é versionar um dataset CSV e rodar um pipeline com DVC, registrar execuções no MLflow e comparar resultados. O laboratório usa `data/avaliacoes.csv`, um conjunto sintético para fins didáticos. O modelo estima `final_grade` a partir das variáveis de estudo e frequência.

- **Git** registra código, configurações, ponteiros do DVC e o estado do pipeline.
- **DVC** acompanha o conteúdo do CSV, seus hashes, parâmetros e métricas do pipeline.
- **MLflow Tracking** registra execuções, parâmetros, métricas, identificação do conjunto de dados e artefato do modelo.
- O contêiner de treinamento permanece ativo para uso em terminal interativo.

## Estrutura

```text
mlflow/
├── data/avaliacoes.csv #Será criado sinteticamente
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

## 1. Inicializar o Git e o DVC e versionar o CSV

Configure a identificação utilizada nos commits e inicialize o repositório Git do laboratório:

```bash
git config --global user.name "Aluno"
git config --global user.email "aluno@example.org"
git init -b main
git status
```

Crie o conjunto de dados sintético para a prática:

```bash
mkdir -p data
python - <<'PY'
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 120
study_hours = rng.integers(1, 21, n)
attendance_pct = rng.integers(50, 101, n)
previous_grade = rng.integers(30, 101, n)
final_grade = (
    2.0 * study_hours
    + 0.25 * attendance_pct
    + 0.40 * previous_grade
    - 10
    + rng.normal(0, 5, n)
)
pd.DataFrame({
    "study_hours": study_hours,
    "attendance_pct": attendance_pct,
    "previous_grade": previous_grade,
    "final_grade": np.round(np.clip(final_grade, 0, 100), 2),
}).to_csv("data/avaliacoes.csv", index=False)
PY
```

Inicialize o DVC e adicione o CSV:

```bash
dvc init
dvc add data/avaliacoes.csv
```

O Git passa a registrar `data/avaliacoes.csv.dvc`, que contém o identificador da versão; o conteúdo fica no cache local do DVC. O CSV permanece disponível em `data/avaliacoes.csv` e é ignorado pelo Git.

Registre a versão inicial:

```bash
git add .
git commit -m "Base inicial e dados versionados com DVC"
```

## 2. Executar o pipeline

```bash
dvc repro
dvc metrics show
git status
git add dvc.lock metrics.json
git commit -m "Registra execução de referência"
```

O DVC lê os parâmetros de `params.yaml`, executa `src/train.py` e atualiza `metrics.json`. O treinamento também envia parâmetros, métricas e o modelo ao experimento `avaliacoes-academicas` no MLflow.

O `train.py` utiliza `skops_trusted_types=["sklearn.tree._tree.Tree"]` em `mlflow.sklearn.log_model` para permitir a serialização do modelo Random Forest.

## 3. Comparar duas execuções

Altere `n_estimators` de `30` para `100` em `params.yaml` e rode:

```bash
python - <<'PY'
from pathlib import Path
p = Path("params.yaml")
s = p.read_text().replace("n_estimators: 30", "n_estimators: 100")
p.write_text(s)
PY
dvc repro
dvc metrics diff HEAD
```

No MLflow, compare as execuções. Observe os parâmetros, `mae`, `rmse`, `r2`, `dataset_sha256` e o artefato do modelo. O hash do CSV deve permanecer igual; nesta comparação, o parâmetro alterado foi o número de árvores.

Exemplo ilustrativo (os valores dependerão do dataset e da execução):

| Métrica | 30 árvores | 100 árvores |
|---|---:|---:|
| MAE | 4,1416 | 4,1112 |
| RMSE | 5,3607 | 5,3033 |
| R² | 0,8695 | 0,8723 |

A melhora é pequena neste exemplo e vale para essa divisão dos dados. Uma execução não demonstra que 100 árvores sejam sempre melhores.

Registre a segunda execução:

```bash
git add params.yaml dvc.lock metrics.json
git commit -m "Compara modelo com 100 árvores"
```

## 4. Versionar o dataset

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

Para restaurar a versão anterior, recupere o ponteiro do commit imediatamente anterior à alteração do CSV:

```bash
git restore --source=HEAD~1 -- data/avaliacoes.csv.dvc
dvc checkout
dvc repro
```

O Git recupera o ponteiro versionado; o `dvc checkout` materializa o CSV correspondente a partir do cache local. O `dvc repro` atualiza o treinamento para os dados restaurados.

## Parar os serviços

Saia do contêiner:

```bash
exit
```

Na pasta `mlflow` do computador, execute:

```bash
docker compose down
```

O banco e os artefatos do MLflow permanecem no volume nomeado. Para apagá-los também:

```bash
docker compose down -v
```

## Considerações finais

Neste laboratório, vimos como utilizar Git, DVC e MLflow desempenham funções complementares no gerenciamento de experimentos de Machine Learning.

O **Git** permitiu preservar o histórico do código e das configurações. O **DVC** acrescentou o controle das versões dos dados e das dependências do pipeline, enquanto o **MLflow** registrou os experimentos, seus parâmetros, métricas e modelos.

A comparação dos treinamentos mostrou que o aumento de 30 para 100 árvores no Random Forest não produziu necessariamente uma melhoria no desempenho do modelo. 

Posteriormente, a alteração do conjunto de dados também modificou os resultados, evidenciando que tanto os parâmetros quanto os dados influenciam o comportamento do modelo.

Também trabalhamos a recuperação de uma versão anterior do dataset. O DVC identificou que o treinamento correspondente já havia sido realizado e recuperou os resultados armazenados em cache, sem necessidade de novo processamento. 

Assim, vimos como a reprodutibilidade envolve não apenas repetir uma execução, mas também preservar as condições e os resultados que permitem reconstruir o histórico dos experimentos.

A principal contribuição dessas ferramentas é permitir que cada resultado de Machine Learning seja associado aos dados, ao código e aos parâmetros que o produziram. Dessa forma, o desenvolvimento deixa de depender de experimentos isolados e passa a contar com um processo organizado, rastreável e passível de reprodução.

<!--
Providenciar MinIO/S3 para um remoto DVC e GitLab local para hospedagem e CI/CD.
1. O que o Git guarda sobre o CSV e onde fica seu conteúdo?
2. Qual é a diferença entre o papel do DVC e o do MLflow?
3. Como identificar quais dados e parâmetros produziram um modelo?
4. O que mudou entre as execuções de 30 e 100 árvores?
-->
