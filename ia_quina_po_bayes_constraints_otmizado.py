import numpy as np
import pandas as pd
import requests
import io
import math
import random
import os
from collections import Counter
from datetime import datetime
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

# === CONFIGURAÇÃO DO AMBIENTE ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO MOTOR: MARKOV SNIPER + RESTRIÇÃO ANTI-SEQUÊNCIA BRUTAL")
print("=" * 85)

# === DOWNLOAD DE DADOS ===
print("Baixando dados oficiais da Quina diretamente da Caixa...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df_quina = pd.read_excel(io.BytesIO(response.content))

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
df_cronologico = df_quina.sort_values(by='Concurso').copy().reset_index(drop=True)
total_sorteios = len(df_cronologico)

# =============================================================
# 1. ENGENHARIA DE PROVIMENTO: INDEXAÇÃO MATEMÁTICA PURA O(1)
# =============================================================
def jogo_para_indice(jogo):
    idx = 1
    atual = 1
    for r, bola in enumerate(jogo):
        for v in range(atual, bola):
            idx += math.comb(80 - v, 5 - 1 - r)
        atual = bola + 1
    return idx

print("Mapeando a série temporal de índices combinatórios...")
df_cronologico['Indice_Combinacao'] = [jogo_para_indice(sorted([int(row[c]) for c in colunas_bolas])) for _, row in df_cronologico.iterrows()]

# =============================================================
# 2. MATRIZ DE TRANSIÇÃO DE ESTADOS (MARKOV)
# =============================================================
transicao_contagem = Counter()
jejum_total = Counter()
atraso_atual = {d: 0 for d in range(1, 81)}

for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    for d in range(1, 81):
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
# 3. [NOVA RESTRIÇÃO] PROIBIÇÃO DE SEQUÊNCIAS (MÁXIMO 2 COLADOS)
# =============================================================
def validar_anti_sequencia(jogo):
    """
    Retorna False se o jogo possuir 3 ou mais números seguidos.
    Ex: [10, 11, 12, 40, 50] -> Bloqueado (False)
    Ex: [01, 02, 15, 16, 40] -> Permitido (True) - Possui apenas duplas seguidas.
    """
    for i in range(len(jogo) - 2):
        if jogo[i+1] == jogo[i] + 1 and jogo[i+2] == jogo[i] + 2:
            return False
    return True

# =============================================================
# 4. AVALIADOR DE COERÊNCIA DE ÍNDICES CONSECUTIVOS
# =============================================================
def avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2):
    if idx_n1 == 0 or idx_n2 == 0: 
        return True
    idx_atual = jogo_para_indice(jogo)
    if abs(idx_atual - idx_n1) < 100000: 
        return False
    salto_anterior = idx_n1 - idx_n2
    salto_atual_proposto = idx_atual - idx_n1
    if abs(salto_atual_proposto - salto_anterior) < 25000: 
        return False
    return True

# =============================================================
# 5. SOLVER MIP COM ADAPTAÇÃO DINÂMICA DE CORTES E FILTROS FÍSICOS
# =============================================================
def solver_markov_mip(atrasos_atuais, idx_n1, idx_n2, n_jogos=5):
    scores = {d: prob_sair(atrasos_atuais[d-1]) for d in range(1, 81)}
    
    jogos = []
    for rodada in range(n_jogos):
        prob = LpProblem(f"Markov_Quina_Ousado_{rodada}", LpMaximize)
        x = LpVariable.dicts("d", range(1, 81), cat="Binary")
        
        prob += lpSum((scores[d] + (random.random() * 0.1)) * x[d] for d in range(1, 81))
        
        prob += lpSum(x[d] for d in range(1, 81)) == 5
        prob += lpSum(x[d] for d in range(1, 81) if d % 2 == 0) >= 2
        prob += lpSum(x[d] for d in range(1, 81) if d % 2 != 0) >= 2
        
        prob += lpSum(d * x[d] for d in range(1, 81)) >= 130
        prob += lpSum(d * x[d] for d in range(1, 81)) <= 270
        
        for jogo_previo in jogos:
            prob += lpSum(x[d] for d in jogo_previo) <= 2
            
        tentativas_cut = 0
        while tentativas_cut < 25: # Aumentado limite de busca devido à nova restrição
            prob.solve(PULP_CBC_CMD(msg=False))
            if LpStatus[prob.status] == "Optimal":
                jogo_proposto = sorted([d for d in range(1, 81) if x[d].varValue == 1])
                
                # Aplica conjuntamente a barreira de índices e a barreira de anti-sequência física
                if validar_anti_sequencia(jogo_proposto) and avaliar_coerencia_indices_fuzzy(jogo_proposto, idx_n1, idx_n2):
                    jogos.append(jogo_proposto)
                    break
                else:
                    # Injeta corte linear na matriz do solver caso falte simetria ou crie sequências ruins
                    prob += lpSum(x[d] for d in jogo_proposto) <= 3
                    tentativas_cut += 1
            else:
                break
                
        if len(jogos) <= rodada:
            # Fallback seguro caso os filtros asfixiem o solver MIP
            jogo_seguro = sorted(random.sample(range(1, 81), 5))
            while not validar_anti_sequencia(jogo_seguro):
                jogo_seguro = sorted(random.sample(range(1, 81), 5))
            jogos.append(jogo_seguro)
            
    return jogos

# =============================================================
# 6. BACKTESTING DE VERIFICAÇÃO HISTÓRICA
# =============================================================
print("\n[FASE 2] Executando Backtesting com Barreiras Ativas (Índices + Anti-Sequência)...")
split_idx = int(len(df_cronologico) * 0.8)
total_duques, total_ternos, total_quadras, total_quinas = 0, 0, 0, 0
total_bilhetes = 0

for i in range(split_idx, len(df_cronologico)):
    df_passado = df_cronologico.iloc[:i]
    atrasos_ponto = {d: 0 for d in range(1, 81)}
    for _, row in df_passado.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        for d in range(1, 81):
            if d in sorteados: atrasos_ponto[d] = 0
            else: atrasos_ponto[d] += 1
            
    idx_n1_ponto = df_passado.iloc[-1]['Indice_Combinacao'] if len(df_passado) > 0 else 0
    idx_n2_ponto = df_passado.iloc[-2]['Indice_Combinacao'] if len(df_passado) > 1 else 0
            
    jogos = solver_markov_mip([atrasos_ponto[d] for d in range(1, 81)], idx_n1_ponto, idx_n2_ponto, n_jogos=5)
    
    gabarito = set([int(df_cronologico.iloc[i][c]) for c in colunas_bolas])
    for j in jogos:
        total_bilhetes += 1
        acertos = len(set(j) & gabarito)
        if acertos == 2: total_duques += 1
        elif acertos == 3: total_ternos += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 5: total_quinas += 1

total_bilhetes = max(1, total_bilhetes)

# =============================================================
# 7. PALPITES FINAIS PARA O PRÓXIMO CONCURSO
# =============================================================
atraso_hoje = {d: 0 for d in range(1, 81)}
for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    for d in range(1, 81):
        if d in sorteados: atraso_hoje[d] = 0
        else: atraso_hoje[d] += 1

idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

palpites = solver_markov_mip([atraso_hoje[d] for d in range(1, 81)], idx_n1_real, idx_n2_real, n_jogos=5)

# =============================================================
# 8. GRAVAÇÃO COMPACTA NO HISTÓRICO TXT
# =============================================================
data_hora_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

relatorio_txt = f"""
=====================================================================================
TESTE EXECUTADO EM: {data_hora_atual}
=====================================================================================
MODELO: MIP Sniper + Restrição de Índices + NOVA BARREIRA ANTI-SEQUÊNCIA (Max 2 colados)
CONFIGURAÇÃO DO SOLVER: Paridade Estrita, Soma (130-270), Anti-Repetição (<=2)
TOTAL DE BILHETES AUDITADOS NO BACKTESTING: {total_bilhetes}

RESULTADOS EM PERCENTUAIS:
▶ DUQUES:  {total_duques} cartões ➔ {(total_duques / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 2.7661%)
▶ TERNOS:  {total_ternos} cartões ➔ {(total_ternos / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.1152%)
▶ QUADRAS: {total_quadras} cartões ➔ {(total_quadras / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0016%)
▶ QUINAS:  {total_quinas} cartões ➔ {(total_quinas / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0000%)

PALPITES EMITIDOS PARA O PRÓXIMO CONCURSO REAL:
"""

for i, j in enumerate(palpites, 1):
    relatorio_txt += f"Jogo {i}: {j} | Índice Real: {jogo_para_indice(j):,} | Soma: {sum(j)}\n"

relatorio_txt += "=====================================================================================\n\n"

print(relatorio_txt)

nome_arquivo = "historico_testes_quina.txt"
with open(nome_arquivo, "a", encoding="utf-8") as f:
    f.write(relatorio_txt)

print(f"-> Concluído! Resultados adicionados ao arquivo histórico para comparação.")