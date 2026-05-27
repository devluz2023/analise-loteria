import numpy as np
import pandas as pd
import requests
import io
import math
import os
from collections import Counter
from datetime import datetime
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO MOTOR PRO: MARKOV DINÂMICO + MATRIZ DE CO-OCORRÊNCIA")
print("=" * 85)

# === DOWNLOAD DE DADOS ===
print("Baixando dados oficiais da Caixa...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df_quina = pd.read_excel(io.BytesIO(response.content))

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
df_cronologico = df_quina.sort_values(by='Concurso').copy().reset_index(drop=True)
total_sorteios = len(df_cronologico)

# =============================================================
# 1. INDEXAÇÃO MATEMÁTICA PURA O(1)
# =============================================================
def jogo_para_indice(jogo):
    idx = 1
    atual = 1
    for r, bola in enumerate(jogo):
        for v in range(atual, bola):
            idx += math.comb(80 - v, 5 - 1 - r)
        atual = bola + 1
    return idx

df_cronologico['Indice_Combinacao'] = [jogo_para_indice(sorted([int(row[c]) for c in colunas_bolas])) for _, row in df_cronologico.iterrows()]

# =============================================================
# 2. CONFIGURAÇÃO DE PARAMETRIZAÇÃO AVANÇADA (MELHORIA 1)
# =============================================================
JANELA_MEMORIA_MARKOV = 1500  # Só estuda os últimos X concursos para evitar dados obsoletos

def calcular_probabilidades_markov(df_subset):
    transicao_contagem = Counter()
    jejum_total = Counter()
    atraso_atual = {d: 0 for d in range(1, 81)}

    for index, row in df_subset.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        for d in range(1, 81):
            jejum_anterior = atraso_atual[d]
            jejum_total[jejum_anterior] += 1
            if d in sorteados:
                transicao_contagem[(jejum_anterior, 0)] += 1
                atraso_atual[d] = 0
            else:
                atraso_atual[d] += 1
                
    probs = {}
    for d in range(1, 81):
        j_atual = atraso_atual[d]
        total = jejum_total.get(j_atual, 1)
        saiu = transicao_contagem.get((j_atual, 0), 0)
        probs[d] = saiu / total
    return probs, atraso_atual

# =============================================================
# 3. CONSTRUÇÃO DA MATRIZ DE CO-OCORRÊNCIA (MELHORIA 2)
# =============================================================
print("Mapeando afinidades de dezenas (Co-ocorrência)...")
matriz_coocorrencia = np.zeros((81, 81), dtype=np.float32)
for index, row in df_cronologico.iterrows():
    sorteados = [int(row[c]) for c in colunas_bolas]
    for d1 in sorteados:
        for d2 in sorteados:
            if d1 != d2:
                matriz_coocorrencia[d1, d2] += 1

# Normaliza a matriz para virar um score de afinidade (0 a 1)
if matriz_coocorrencia.max() > 0:
    matriz_coocorrencia /= matriz_coocorrencia.max()

# =============================================================
# 4. FILTROS E VALIDAÇÕES GEOMÉTRICAS
# =============================================================
def validar_anti_sequencia(jogo):
    for i in range(len(jogo) - 2):
        if jogo[i+1] == jogo[i] + 1 and jogo[i+2] == jogo[i] + 2:
            return False
    return True

def avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2):
    if idx_n1 == 0 or idx_n2 == 0: return True
    idx_atual = jogo_para_indice(jogo)
    if abs(idx_atual - idx_n1) < 100000: return False
    if abs((idx_atual - idx_n1) - (idx_n1 - idx_n2)) < 25000: return False
    return True

# =============================================================
# 5. SOLVER MIP COMBINADO (MARKOV + AFINIDADE DE CO-OCORRÊNCIA)
# =============================================================
def solver_markov_coocorrencia_mip(probs_markov, afinidades_vizinhas, idx_n1, idx_n2, n_jogos=5):
    jogos = []
    
    # Identifica as dezenas que estão mais fortes em Markov para guiar a afinidade
    dezenas_ancora = sorted(probs_markov.items(), key=lambda x: x[1], reverse=True)[:10]
    ancoras = [d[0] for d in dezenas_ancora]

    for rodada in range(n_jogos):
        prob = LpProblem(f"MIP_Pro_{rodada}", LpMaximize)
        x = LpVariable.dicts("d", range(1, 81), cat="Binary")
        
        # O OBJETIVO AGORA FUSA: Força de Exaustão (Markov) + Afinidade de Saírem Juntos
        score_afinidade = {d: sum(afinidades_vizinhas[d, a] for a in ancoras) for d in range(1, 81)}
        
        prob += lpSum((probs_markov[d] * 0.7 + score_afinidade[d] * 0.3) * x[d] for d in range(1, 81))
        
        prob += lpSum(x[d] for d in range(1, 81)) == 5
        prob += lpSum(x[d] for d in range(1, 81) if d % 2 == 0) >= 2
        prob += lpSum(x[d] for d in range(1, 81) if d % 2 != 0) >= 2
        
        prob += lpSum(d * x[d] for d in range(1, 81)) >= 130
        prob += lpSum(d * x[d] for d in range(1, 81)) <= 270
        
        for jogo_previo in jogos:
            prob += lpSum(x[d] for d in jogo_previo) <= 2
            
        tentativas_cut = 0
        while tentativas_cut < 25:
            prob.solve(PULP_CBC_CMD(msg=False))
            if LpStatus[prob.status] == "Optimal":
                jogo_proposto = sorted([d for d in range(1, 81) if x[d].varValue == 1])
                if validar_anti_sequencia(jogo_proposto) and avaliar_coerencia_indices_fuzzy(jogo_proposto, idx_n1, idx_n2):
                    jogos.append(jogo_proposto)
                    break
                else:
                    prob += lpSum(x[d] for d in jogo_proposto) <= 3
                    tentativas_cut += 1
            else:
                break
                
        if len(jogos) <= rodada:
            jogos.append(sorted(random.sample(range(1, 81), 5)))
            
    return jogos

# =============================================================
# 6. BACKTESTING PRO
# =============================================================
print("\n[FASE 2] Executando Backtesting Dinâmico...")
split_idx = int(len(df_cronologico) * 0.8)
total_duques, total_ternos, total_quadras, total_quinas = 0, 0, 0, 0
total_bilhetes = 0

for i in range(split_idx, len(df_cronologico)):
    df_passado = df_cronologico.iloc[:i]
    
    # Aplica a melhoria 1: Recorta apenas a janela recente para calcular o Markov
    df_passado_recente = df_passado.iloc[-JANELA_MEMORIA_MARKOV:] if len(df_passado) > JANELA_MEMORIA_MARKOV else df_passado
    
    probs_m, _ = calcular_probabilidades_markov(df_passado_recente)
    
    idx_n1_ponto = df_passado.iloc[-1]['Indice_Combinacao'] if len(df_passado) > 0 else 0
    idx_n2_ponto = df_passado.iloc[-2]['Indice_Combinacao'] if len(df_passado) > 1 else 0
            
    jogos = solver_markov_coocorrencia_mip(probs_m, matriz_coocorrencia, idx_n1_ponto, idx_n2_ponto, n_jogos=5)
    
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
# 7. PALPITES FINAIS
# =============================================================
df_recente = df_cronologico.iloc[-JANELA_MEMORIA_MARKOV:]
probs_finais, _ = calcular_probabilidades_markov(df_recente)

idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

palpites = solver_markov_coocorrencia_mip(probs_finais, matriz_coocorrencia, idx_n1_real, idx_n2_real, n_jogos=5)

# =============================================================
# 8. GRAVAÇÃO NO HISTÓRICO TXT
# =============================================================
data_hora_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

relatorio_txt = f"""
=====================================================================================
TESTE EXECUTADO EM: {data_hora_atual}
=====================================================================================
MODELO: PRO - Markov Dinâmico (Janela {JANELA_MEMORIA_MARKOV}) + Matriz de Co-ocorrência Estrita
CONFIGURAÇÃO: Pesos (70% Markov / 30% Afinidade), Anti-Sequência, Filtro Fuzzy de Índices
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

with open("historico_testes_quina.txt", "a", encoding="utf-8") as f:
    f.write(relatorio_txt)

print("-> Concluído! Verifique o arquivo .txt para comparar a evolução das métricas.")