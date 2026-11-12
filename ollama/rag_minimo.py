"""Laboratório: recuperação semântica + resposta fundamentada com Ollama.

Requer: pip install requests
Modelos: qwen3:4b-instruct-2507-q4_K_M e qwen3-embedding:0.6b
Todos os documentos deste arquivo são fictícios.
"""

import math
import os

import requests

BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
EMBEDDING = os.getenv("OLLAMA_EMBEDDING", "qwen3-embedding:0.6b")

# Corpus fictício de exemplo (não contém registros de ambientes reais).
documentos = [
    "DOC-01 | CMDB: o host lab-app-01 executa a aplicação Sistema de Testes e possui 8 GiB de RAM.",
    "DOC-02 | Configuração JVM: a instância lab-app-01 usa -Xmx4g e coletor G1GC.",
    "DOC-03 | Log de teste: o host lab-app-01 registrou pausas de GC de 850 ms e 1200 ms.",
    "DOC-04 | Operação: não existe diagnóstico conclusivo da causa das pausas nos dados fornecidos.",
]
pergunta = "Qual é o limite de heap da JVM do lab-app-01 e quais evidências existem sobre o GC?"


def post(endpoint, payload):
    resposta = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=240)
    resposta.raise_for_status()
    return resposta.json()


def cosseno(a, b):
    numerador = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    return numerador / (norma_a * norma_b) if norma_a and norma_b else 0.0


# 1. Gerar vetores para os documentos e para a pergunta.
vetores = post("/api/embed", {"model": EMBEDDING, "input": documentos + [pergunta], "truncate": False})["embeddings"]
vetor_pergunta = vetores[-1]

# 2. Ordenar os documentos pela similaridade semântica.
ranking = sorted(
    ((cosseno(vetores[i], vetor_pergunta), doc) for i, doc in enumerate(documentos)),
    reverse=True,
)
selecionados = ranking[:3]

print("DOCUMENTOS RECUPERADOS:")
for pontuacao, documento in selecionados:
    print(f"  similaridade={pontuacao:.4f} | {documento}")

# 3. Fornecer fontes recuperadas ao LLM e exigir referência.
contexto = "\n".join(documento for _, documento in selecionados)
resultado = post(
    "/api/chat",
    {
        "model": LLM,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Responda em português usando apenas as fontes fornecidas. "
                    "Indique os IDs DOC-xx que fundamentam a resposta. "
                    "Se faltar evidência, declare a limitação; não invente causas."
                ),
            },
            {"role": "user", "content": f"FONTES:\n{contexto}\n\nPERGUNTA:\n{pergunta}"},
        ],
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300},
    },
)
print("\nRESPOSTA:\n", resultado["message"]["content"])
print("\nMÉTRICAS:\n", {
    "total_duration_ns": resultado.get("total_duration"),
    "prompt_eval_count": resultado.get("prompt_eval_count"),
    "eval_count": resultado.get("eval_count"),
})
