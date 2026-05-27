import numpy as np
import pandas as pd
import random
import os
import math
import itertools
from collections import Counter
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from get_data import df_mega_sena

# Importação da lógica de índices
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO MOTOR: CADEIAS DE MARKOV + SOLVER MIP (ESTADO DE EXAUSTÃO)")
print("=" * 85)

# =============================================================
# 1. CONSTRUÇÃO DA MATRIZ DE TRANSIÇÃO DE ESTADOS (MARKOV)
# =============================================================
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy().reset_index(drop=True)

# Mapeia a frequência de transição: Quantas vezes Jejum T virou Jejum 0
# A probabilidade de sair dado o Jejum atual é: (Vezes que saiu no Jejum T) / (Total que esteve no Jejum T)
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

# Probabilidade Condicional P(Sai | Jejum=T)
def prob_sair(jejum):
    total = jejum_total.get(jejum, 1)
    saiu = transicao_contagem.get((jejum, 0), 0)
    return saiu / total

# =============================================================
# 2. SOLVER MIP BASEADO EM PROBABILIDADE MARKOVIANA
# =============================================================
def solver_markov_mip(atrasos_atuais, n_jogos=5):
    # Calcula score baseado na probabilidade de transição para 'Saiu'
    scores = {d: prob_sair(atrasos_atuais[d-1]) for d in range(1, 61)}
    
    jogos = []
    for rodada in range(n_jogos):
        prob = LpProblem(f"Markov_MIP_{rodada}", LpMaximize)
        x = LpVariable.dicts("d", range(1, 61), cat="Binary")
        
        # Objetivo: Maximizar a probabilidade combinada de exaustão
        prob += lpSum(scores[d] * x[d] for d in range(1, 61))
        
        # Restrições Físicas
        prob += lpSum(x[d] for d in range(1, 61)) == 6
        prob += lpSum(x[d] for d in range(1, 61) if d % 2 == 0) >= 2
        prob += lpSum(x[d] for d in range(1, 61) if d % 2 != 0) >= 2
        
        # Evita repetição pesada entre jogos
        for jogo_previo in jogos:
            prob += lpSum(x[d] for d in jogo_previo) <= 3
            
        prob.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[prob.status] == "Optimal":
            jogos.append(sorted([d for d in range(1, 61) if x[d].varValue == 1]))
            
    return jogos

# =============================================================
# 3. PROVA DE ACURÁCIA (BACKTESTING MARKOV)
# =============================================================
print("\n[FASE 2] Executando Backtesting com Cadeias de Markov...")
split_idx = int(len(df_cronologico) * 0.8)
total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes = 0

for i in range(split_idx, len(df_cronologico)):
    # Simula o estado de atraso no momento do concurso
    df_passado = df_cronologico.iloc[:i]
    atrasos_ponto = {d: 0 for d in range(1, 61)}
    for _, row in df_passado.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: atrasos_ponto[d] = 0
            else: atrasos_ponto[d] += 1
            
    jogos = solver_markov_mip([atrasos_ponto[d] for d in range(1, 61)], n_jogos=5)
    
    gabarito = set([int(df_cronologico.iloc[i][c]) for c in colunas_bolas])
    for j in jogos:
        total_bilhetes += 1
        acertos = len(set(j) & gabarito)
        if acertos == 3: total_ternos += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 6: total_senas += 1

print(f"Resultados: Ternos: {total_ternos}, Quadras: {total_quadras}")
print("-" * 85)

# =============================================================
# 4. PALPITE PARA O PRÓXIMO CONCURSO
# =============================================================
atraso_hoje = {d: 0 for d in range(1, 61)}
for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    for d in range(1, 61):
        if d in sorteados: atraso_hoje[d] = 0
        else: atraso_hoje[d] += 1

palpites = solver_markov_mip([atraso_hoje[d] for d in range(1, 61)], n_jogos=5)

print("\nJOGOS GERADOS POR CADEIA DE MARKOV (ESTADO DE EXAUSTÃO):")
for i, j in enumerate(palpites, 1):
    print(f"Jogo {i}: {j}")