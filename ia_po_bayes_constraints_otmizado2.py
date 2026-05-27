import numpy as np
import pandas as pd
import random
import os
import math
from collections import Counter
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from get_data import df_mega_sena

# === CONFIGURAÇÃO DO AMBIENTE ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO MOTOR: MARKOV PURO + SOMA GAUSSIANA (ESTADO DE EXAUSTÃO)")
print("=" * 85)

# =============================================================
# 1. CONSTRUÇÃO DA MATRIZ DE TRANSIÇÃO (ESTADOS DE JEJUM)
# =============================================================
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy().reset_index(drop=True)

transicao_contagem = Counter()
jejum_total = Counter()
atraso_atual = {d: 0 for d in range(1, 61)}

for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    for d in range(1, 61):
        jejum_anterior = atraso_atual[d]
        jejum_total[jejum_anterior] += 1
        
        if d in sorteados:
            transicao_contagem[(jejum_anterior, 0)] += 1
            atraso_atual[d] = 0
        else:
            atraso_atual[d] += 1

def prob_sair(jejum):
    total = jejum_total.get(jejum, 1)
    saiu = transicao_contagem.get((jejum, 0), 0)
    return saiu / total

# =============================================================
# 2. SOLVER MIP (ESTRUTURA PURA)
# =============================================================
def solver_markov_puro(atrasos_atuais, n_jogos=5):
    # Calcula a probabilidade de exaustão para cada número
    scores = {d: prob_sair(atrasos_atuais[d-1]) for d in range(1, 61)}
    
    jogos = []
    for rodada in range(n_jogos):
        prob = LpProblem(f"Markov_Puro_{rodada}", LpMaximize)
        x = LpVariable.dicts("d", range(1, 61), cat="Binary")
        
        # Objetivo: Maximizar apenas a probabilidade de exaustão (Markov)
        prob += lpSum(scores[d] * x[d] for d in range(1, 61))
        
        # Regra de estrutura: soma entre 120 e 200 (Curva Gaussiana da Mega)
        prob += lpSum(d * x[d] for d in range(1, 61)) >= 120
        prob += lpSum(d * x[d] for d in range(1, 61)) <= 200
        
        # Regra de volante: 6 dezenas por jogo
        prob += lpSum(x[d] for d in range(1, 61)) == 6
        
        # Evita que todos os jogos sejam muito parecidos
        for jogo_previo in jogos:
            prob += lpSum(x[d] for d in jogo_previo) <= 3
            
        prob.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[prob.status] == "Optimal":
            jogos.append(sorted([d for d in range(1, 61) if x[d].varValue == 1]))
            
    return jogos

# =============================================================
# 3. BACKTESTING HISTÓRICO
# =============================================================
print("\n[FASE 2] Executando Backtesting de Alta Eficiência...")
split_idx = int(len(df_cronologico) * 0.8)
total_ternos, total_quadras = 0, 0

for i in range(split_idx, len(df_cronologico)):
    df_passado = df_cronologico.iloc[:i]
    atrasos_ponto = {d: 0 for d in range(1, 61)}
    for _, row in df_passado.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: atrasos_ponto[d] = 0
            else: atrasos_ponto[d] += 1
            
    jogos = solver_markov_puro([atrasos_ponto[d] for d in range(1, 61)], n_jogos=5)
    
    gabarito = set([int(df_cronologico.iloc[i][c]) for c in colunas_bolas])
    for j in jogos:
        acertos = len(set(j) & gabarito)
        if acertos == 3: total_ternos += 1
        elif acertos == 4: total_quadras += 1

print(f"Desempenho histórico no período de teste: Ternos={total_ternos}, Quadras={total_quadras}")
print("-" * 85)

# =============================================================
# 4. GERAÇÃO DOS JOGOS REAIS
# =============================================================
atraso_hoje = {d: 0 for d in range(1, 61)}
for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    for d in range(1, 61):
        if d in sorteados: atraso_hoje[d] = 0
        else: atraso_hoje[d] += 1

palpites = solver_markov_puro([atraso_hoje[d] for d in range(1, 61)], n_jogos=5)

print("\nJOGOS DEFINITIVOS GERADOS POR EXAUSTÃO MARKOVIANA:")
for i, j in enumerate(palpites, 1):
    print(f"Jogo {i}: {j} | Soma: {sum(j)}")
print("=" * 85)