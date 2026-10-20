from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "avaliacoes.csv"
df = pd.read_csv(DATA_PATH)
df.loc[:9, "study_hours"] = (df.loc[:9, "study_hours"] + 2).clip(upper=20)
df.to_csv(DATA_PATH, index=False)
print("Alteradas as horas de estudo das dez primeiras linhas.")
