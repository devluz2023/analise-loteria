import numpy as np
import pandas as pd
import requests
import io
import math
import itertools
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 0. CARREGAMENTO E CRIAÇÃO DO ÍNDICE (CORRIGIDO)
# =============================================================
print("Conectando ao serviço da Caixa...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df_quina = pd.read_excel(io.BytesIO(response.content))
df_cronologico = df_quina.sort_values(by='Concurso').copy().reset_index(drop=True)
bolas_cols = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']

def jogo_para_indice(jogo):
    idx = 1
    atual = 1
    for r, bola in enumerate(sorted(jogo)):
        for v in range(atual, bola):
            idx += math.comb(80 - v, 5 - 1 - r)
        atual = bola + 1
    return idx

# CRIAÇÃO FORÇADA DA COLUNA 'Indice'
print("Indexando dados históricos...")
df_cronologico['Indice'] = [jogo_para_indice([int(row[c]) for c in bolas_cols]) for _, row in df_cronologico.iterrows()]

# =============================================================
# 1. MOTORES DE ANÁLISE
# =============================================================
def calcular_probabilidade_salto(df):
    # Verificação de segurança
    if 'Indice' not in df.columns:
        raise KeyError("A coluna 'Indice' não foi encontrada no DataFrame.")
    deltas = df['Indice'].diff().dropna()
    freq = Counter(deltas)
    total = len(deltas)
    return {delta: count / total for delta, count in freq.items()}

def calcular_perfil_dispersao_log(jogo):
    jogo_s = sorted(jogo)
    gaps = [jogo_s[i+1] - jogo_s[i] for i in range(len(jogo_s)-1)]
    return np.log1p(gaps)

def calcular_campo_energia(df_subset):
    p1 = Counter()
    for _, row in df_subset.iloc[-1000:].iterrows():
        for c in bolas_cols: p1[int(row[c])] += 1
    return {d: np.log1p(p1.get(d, 0) * 1000) for d in range(1, 81)}

def filtrar_por_interseccao(pool, sorteio_anterior, min_rep=1):
    sorteio_set = set(sorteio_anterior)
    return [j for j in pool if len(set(j) & sorteio_set) >= min_rep]

# =============================================================
# 2. AUDITORIA E EXECUÇÃO
# =============================================================
prob_saltos = calcular_probabilidade_salto(df_cronologico)
score = calcular_campo_energia(df_cronologico)

def rodar_auditoria(df, prob_saltos):
    print("[RODANDO AUDITORIA...]")
    resultados = {'Duque': 0, 'Terno': 0, 'Quadra': 0, 'Quina': 0}
    erros_emi = []
    total_eventos = 500
    
    # Calcular perfil ideal de todo o histórico
    perfis = [calcular_perfil_dispersao_log([int(row[c]) for c in bolas_cols]) for _, row in df.iterrows()]
    perfil_ideal = np.mean(perfis, axis=0)

    for i in range(len(df) - total_eventos, len(df)):
        df_passado = df.iloc[:i]
        score_p = calcular_campo_energia(df_passado)
        idx_atual = df_passado.iloc[-1]['Indice']
        sorteio_ant = [int(df_passado.iloc[-1][c]) for c in bolas_cols]
        
        elite_15 = sorted([d for d, s in sorted(score_p.items(), key=lambda x: x[1], reverse=True)[:15]])
        pool = filtrar_por_interseccao(list(itertools.combinations(elite_15, 5)), sorteio_ant)
        
        if not pool: continue
        
        # Seleção do melhor palpite com Markov + Log
        melhor = max(pool, key=lambda c: (sum(score_p[d] for d in c) / (np.linalg.norm(calcular_perfil_dispersao_log(c) - perfil_ideal) + 0.1)) * np.log1p(prob_saltos.get(jogo_para_indice(c) - idx_atual, 0.000001) * 10000))
        
        gabarito = [int(df.iloc[i][c]) for c in bolas_cols]
        erros_emi.append(abs(jogo_para_indice(melhor) - jogo_para_indice(gabarito)))
        acertos = len(set(melhor) & set(gabarito))
        
        if acertos == 2: resultados['Duque'] += 1
        elif acertos == 3: resultados['Terno'] += 1
        elif acertos == 4: resultados['Quadra'] += 1
        elif acertos == 5: resultados['Quina'] += 1
            
    return resultados, np.mean(erros_emi), total_eventos

acuracia, emi, total = rodar_auditoria(df_cronologico, prob_saltos)

# Output de resultados
print(f"\n=== RELATÓRIO DE ACURÁCIA ===")
acaso = {'Duque': 2.76, 'Terno': 0.11, 'Quadra': 0.0038, 'Quina': 0.000012}
for p, q in acuracia.items():
    taxa = (q/total)*100
    print(f"▶ {p.upper()}: {taxa:.2f}% | Acaso: {acaso[p]:.3f}% | Delta: {taxa - acaso[p]:+.2f}%")
print(f"➔ ERRO MÉDIO DE ÍNDICE (EMI): {emi:,.0f} | Precisão: {100 - (emi/24040016)*100:.2f}%")

# Palpites Finais
idx_atual = df_cronologico.iloc[-1]['Indice']
sorteio_anterior = [int(df_cronologico.iloc[-1][c]) for c in bolas_cols]
pool = filtrar_por_interseccao(list(itertools.combinations(sorted([d for d, s in sorted(score.items(), key=lambda x: x[1], reverse=True)[:15]]), 5)), sorteio_anterior)
palpites = sorted(pool, key=lambda c: (sum(score[d] for d in c) / (np.linalg.norm(calcular_perfil_dispersao_log(c) - np.mean([calcular_perfil_dispersao_log([int(row[col]) for col in bolas_cols]) for _, row in df_cronologico.iterrows()], axis=0) + 0.1))) * np.log1p(prob_saltos.get(jogo_para_indice(c) - idx_atual, 0.000001) * 10000), reverse=True)[:3]

print(f"\n=== PALPITES SUPREMOS ===")
for i, p in enumerate(palpites, 1):
    print(f"➔ PALPITE {i}: {sorted(list(p))} | Índice: {jogo_para_indice(p):,}")
print("=" * 85)