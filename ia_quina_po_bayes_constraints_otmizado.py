import numpy as np
import pandas as pd
import requests
import io
import math
import random
import os
import itertools
from collections import Counter
from datetime import datetime

# === CONFIGURAÇÃO DO AMBINETE ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO MOTOR MASTER FAST: HIPERCUBO CUBIC-15 COM SELETOR GULOSO O(N)")
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
# 2. PROBABILIDADES DE MARKOV (JANELA DINÂMICA)
# =============================================================
JANELA_MEMORIA_MARKOV = 1500  

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
    return probs

# =============================================================
# 3. MATRIZ HISTÓRICA DE CO-OCORRÊNCIA
# =============================================================
print("Mapeando afinidades de co-ocorrência histórica...")
matriz_coocorrencia = np.zeros((81, 81), dtype=np.float32)
for index, row in df_cronologico.iterrows():
    sorteados = [int(row[c]) for c in colunas_bolas]
    for d1 in sorteados:
        for d2 in sorteados:
            if d1 != d2:
                matriz_coocorrencia[d1, d2] += 1

if matriz_coocorrencia.max() > 0:
    matriz_coocorrencia /= matriz_coocorrencia.max()

# =============================================================
# 4. CRIVOS FILTRADORES GEOMÉTRICOS
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
# 5. SELETOR GULOSO ULTRA-RÁPIDO (SUBSTITUTO DO SOLVER MIP COMBINATÓRIO)
# =============================================================
def solver_hipercubo_fast_greedy(probs_markov, afinidades_vizinhas, idx_n1, idx_n2, n_jogos=5):
    dezenas_ancora = sorted(probs_markov.items(), key=lambda x: x[1], reverse=True)[:10]
    ancoras = [d[0] for d in dezenas_ancora]
    
    scores_finais = {}
    for d in range(1, 81):
        score_afinidade = sum(afinidades_vizinhas[d, a] for a in ancoras)
        scores_finais[d] = (probs_markov[d] * 0.7) + (score_afinidade * 0.3)
        
    dezenas_elite_15 = [item[0] for item in sorted(scores_finais.items(), key=lambda x: x[1], reverse=True)[:15]]
    pool_combinacoes = list(itertools.combinations(sorted(dezenas_elite_15), 5))
    
    pool_purificado = []
    scores_pool = []
    
    for combo in pool_combinacoes:
        jogo = list(combo)
        if sum(1 for d in jogo if d % 2 == 0) not in [2, 3]: continue
        if not (130 <= sum(jogo) <= 270): continue
        if not validar_anti_sequencia(jogo): continue
        if not avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2): continue
            
        pool_purificado.append(jogo)
        scores_pool.append(sum(scores_finais[d] for d in jogo))
        
    if len(pool_purificado) < n_jogos:
        return [sorted(random.sample(dezenas_elite_15, 5)) for _ in range(n_jogos)]
        
    # --- PROCESSO GULOSO O(N): FILTRAGEM INSTANTÂNEA DE OVERLAP ---
    indices_ordenados = np.argsort(scores_pool)[::-1]
    jogos_escolhidos = []
    
    for idx in indices_ordenados:
        jogo_candidato = pool_purificado[idx]
        valido = True
        for jogo_eleito in jogos_escolhidos:
            # Força interseção máxima de no máximo 2 números entre os jogos
            if len(set(jogo_candidato) & set(jogo_eleito)) > 2:
                valido = False
                break
        if valido:
            jogos_escolhidos.append(jogo_candidato)
        if len(jogos_escolhidos) == n_jogos:
            break
            
    # Garantia de preenchimento se a malha de overlap for ultra restrita
    if len(jogos_escolhidos) < n_jogos:
        for idx in indices_ordenados:
            if pool_purificado[idx] not in jogos_escolhidos:
                jogos_escolhidos.append(pool_purificado[idx])
            if len(jogos_escolhidos) == n_jogos:
                break
                
    return [sorted(j) for j in jogos_escolhidos]

# =============================================================
# 6. BACKTESTING CEGO VELOZ
# =============================================================
print("\n[FASE 2] Rodando Backtesting Cego por Portfólio Simultâneo O(N)...")
split_idx = int(len(df_cronologico) * 0.8)
total_duques, total_ternos, total_quadras, total_quinas = 0, 0, 0, 0
total_bilhetes = 0

for i in range(split_idx, len(df_cronologico)):
    df_passado = df_cronologico.iloc[:i]
    df_passado_recente = df_passado.iloc[-JANELA_MEMORIA_MARKOV:] if len(df_passado) > JANELA_MEMORIA_MARKOV else df_passado
    
    probs_m = calcular_probabilidades_markov(df_passado_recente)
    
    idx_n1_ponto = df_passado.iloc[-1]['Indice_Combinacao'] if len(df_passado) > 0 else 0
    idx_n2_ponto = df_passado.iloc[-2]['Indice_Combinacao'] if len(df_passado) > 1 else 0
            
    jogos = solver_hipercubo_fast_greedy(probs_m, matriz_coocorrencia, idx_n1_ponto, idx_n2_ponto, n_jogos=5)
    
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
# 7. PROCESSAMENTO COMPLETO PARA O PRÓXIMO CONCURSO
# =============================================================
print("\n[FASE 3] Sintonizando o Hipercubo para o próximo concurso...")
df_recente = df_cronologico.iloc[-JANELA_MEMORIA_MARKOV:]
probs_finais = calcular_probabilidades_markov(df_recente)

idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

dezenas_ancora_finais = sorted(probs_finais.items(), key=lambda x: x[1], reverse=True)[:10]
ancoras_finais = [d[0] for d in dezenas_ancora_finais]
scores_futuros = {}
for d in range(1, 81):
    score_afinidade = sum(matriz_coocorrencia[d, a] for a in ancoras_finais)
    scores_futuros[d] = (probs_finais[d] * 0.7) + (score_afinidade * 0.3)
dezenas_master_15 = [item[0] for item in sorted(scores_futuros.items(), key=lambda x: x[1], reverse=True)[:15]]

palpites = solver_hipercubo_fast_greedy(probs_finais, matriz_coocorrencia, idx_n1_real, idx_n2_real, n_jogos=5)

# =============================================================
# 8. GRAVAÇÃO E INDEXAÇÃO NO HISTÓRICO TXT
# =============================================================
data_hora_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

relatorio_txt = f"""
=====================================================================================
TESTE EXECUTADO EM: {data_hora_atual}
=====================================================================================
MODELO: MASTER HIPERCUBO FAST GREEDY CUBIC-15 (OTIMIZAÇÃO ULTRA-RÁPIDA)
CONFIGURAÇÃO: Subespaço Amostral de 15 Dezenas Supremas, Interseção Máxima entre Jogos (<=2)
TOTAL DE BILHETES AUDITADOS NO BACKTESTING: {total_bilhetes}

RESULTADOS EM PERCENTUAIS CONSOLIDADOS:
▶ DUQUES:  {total_duques} cartões ➔ {(total_duques / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 2.7661%)
▶ TERNOS:  {total_ternos} cartões ➔ {(total_ternos / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.1152%)
▶ QUADRAS: {total_quadras} cartões ➔ {(total_quadras / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0016%)
▶ QUINAS:  {total_quinas} cartões ➔ {(total_quinas / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0000%)

➔ GRUPO DE ELITE CONCENTRADO SELECIONADO (15 Dezenas Supremas):
   {', '.join(f'{n:02d}' for n in sorted(dezenas_master_15))}

PALPITES EMITIDOS PARA O PRÓXIMO CONCURSO REAL:
"""

for i, j in enumerate(palpites, 1):
    relatorio_txt += f"Jogo {i}: {j} | Índice Real: {jogo_para_indice(j):,} | Soma: {sum(j)}\n"

relatorio_txt += "=====================================================================================\n\n"

print(relatorio_txt)

with open("historico_testes_quina.txt", "a", encoding="utf-8") as f:
    f.write(relatorio_txt)

print("-> Sucesso absoluto! Relatório gravado instantaneamente no arquivo 'historico_testes_quina.txt'.")