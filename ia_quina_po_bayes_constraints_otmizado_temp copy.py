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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  MOTOR SNIPER DEFINITIVO: EXTRAÇÃO DE 1 VOLANTE SUPREMO (ALVO ÚNICO)")
print("=" * 85)

# === DOWNLOAD DE DADOS ===
print("Baixando dados oficiais da Quina diretamente da Caixa...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df_quina = pd.read_excel(io.BytesIO(response.content))

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
df_cronologico = df_quina.sort_values(by='Concurso').copy().reset_index(drop=True)

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
# 3. MATRIZ HISTÓRICA DE CO-OCORRÊNCIA (GRAFO DE AFINIDADE)
# =============================================================
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
# 4. CRIVOS PURIFICADORES GEOMÉTRICOS RÍGIDOS
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
# 5. EXTRATOR DO SELETOR DE PRECISÃO DE 1 VOLANTE
# =============================================================
def solver_sniper_1volante(probs_markov, afinidades_vizinhas, idx_n1, idx_n2):
    # Seleciona o grupo estrutural de 15 dezenas de rede
    candidatas_markov = [item[0] for item in sorted(probs_markov.items(), key=lambda x: x[1], reverse=True)[:26]]
    
    melhor_score_subgrafo = -1
    dezenas_elite_15 = []
    
    for _ in range(1000):
        subgrupo_teste = random.sample(candidatas_markov, 15)
        score_conexao = 0
        for d1, d2 in itertools.combinations(subgrupo_teste, 2):
            score_conexao += afinidades_vizinhas[d1, d2]
        score_total_rede = score_conexao + sum(probs_markov[d] * 2.5 for d in subgrupo_teste)
        
        if score_total_rede > melhor_score_subgrafo:
            melhor_score_subgrafo = score_total_rede
            dezenas_elite_15 = sorted(subgrupo_teste)

    # Varre as 3.003 combinações do espaço amostral completo
    pool_combinacoes = list(itertools.combinations(dezenas_elite_15, 5))
    pool_purificado = []
    scores_pool = []
    
    for combo in pool_combinacoes:
        jogo = list(combo)
        
        # Purificação rigorosa por filtros analíticos
        if sum(1 for d in jogo if d % 2 == 0) not in [2, 3]: continue
        if not (130 <= sum(jogo) <= 270): continue
        if not validar_anti_sequencia(jogo): continue
        if not avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2): continue
            
        pool_purificado.append(jogo)
        
        # O score do bilhete mede o equilíbrio dinâmico de exaustão e acoplamento de pares
        score_bilhete = sum(probs_markov[d] for d in jogo) + sum(afinidades_vizinhas[d1, d2] for d1, d2 in itertools.combinations(jogo, 2))
        scores_pool.append(score_bilhete)
        
    if len(pool_purificado) == 0:
        return sorted(random.sample(dezenas_elite_15, 5)), dezenas_elite_15
        
    # EXTRAÇÃO CIRÚRGICA: Localiza a combinação com maior índice energético puro do bloco
    idx_maximo_energia = np.argmax(scores_pool)
    volante_supremo = pool_purificado[idx_maximo_energia]
                    
    return sorted(volante_supremo), dezenas_elite_15

# =============================================================
# 6. BACKTESTING HISTÓRICO DE AUDITORIA DE 1 VOLANTE
# =============================================================
print("\n[FASE 2] Rodando Backtesting de Precisão Cirúrgica (Apenas 1 Volante por Concurso)...")
split_idx = int(len(df_cronologico) * 0.8)
total_duques, total_ternos, total_quadras, total_quinas = 0, 0, 0, 0
total_bilhetes = 0

for i in range(split_idx, len(df_cronologico)):
    df_passado = df_cronologico.iloc[:i]
    df_passado_recente = df_passado.iloc[-JANELA_MEMORIA_MARKOV:] if len(df_passado) > JANELA_MEMORIA_MARKOV else df_passado
    probs_m = calcular_probabilidades_markov(df_passado_recente)
    
    idx_n1_ponto = df_passado.iloc[-1]['Indice_Combinacao'] if len(df_passado) > 0 else 0
    idx_n2_ponto = df_passado.iloc[-2]['Indice_Combinacao'] if len(df_passado) > 1 else 0
            
    # Extrai o único volante candidato para aquele concurso específico
    jogo_unico, _ = solver_sniper_1volante(probs_m, matriz_coocorrencia, idx_n1_ponto, idx_n2_ponto)
    
    gabarito = set([int(df_cronologico.iloc[i][c]) for c in colunas_bolas])
    total_bilhetes += 1
    acertos = len(set(jogo_unico) & gabarito)
    if acertos == 2: total_duques += 1
    elif acertos == 3: total_ternos += 1
    elif acertos == 4: total_quadras += 1
    elif acertos == 5: total_quinas += 1

total_bilhetes = max(1, total_bilhetes)

# =============================================================
# 7. PROJEÇÃO DO PRÓXIMO CONCURSO REAL
# =============================================================
print("\n[FASE 3] Sintonizando e extraindo o palpite definitivo...")
df_recente = df_cronologico.iloc[-JANELA_MEMORIA_MARKOV:]
probs_finais = calcular_probabilidades_markov(df_recente)

idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

palpite_definitivo, dezenas_master_15 = solver_sniper_1volante(probs_finais, matriz_coocorrencia, idx_n1_real, idx_n2_real)

# =============================================================
# 8. GRAVAÇÃO COMPACTA NO HISTÓRICO TXT
# =============================================================
data_hora_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

relatorio_txt = f"""
=====================================================================================
TESTE EXECUTADO EM: {data_hora_atual}
=====================================================================================
MODELO: MONSTRO SNIPER - EXTRAÇÃO DE 1 VOLANTE DE MÁXIMA ENERGIA COMBINATÓRIA
CONFIGURAÇÃO: Varredura de todo o espaço Cubic-15 com isolamento de Linha Áurea Única
TOTAL DE BILHETES AUDITADOS NO BACKTESTING (1 POR CONCURSO): {total_bilhetes}

RESULTADOS CONSOLIDADOS DO PORTFÓLIO DE VOLANTE ÚNICO:
▶ DUQUES:  {total_duques} cartões ➔ {(total_duques / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 2.7661%)
▶ TERNOS:  {total_ternos} cartões ➔ {(total_ternos / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.1152%)
▶ QUADRAS: {total_quadras} cartões ➔ {(total_quadras / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0016%)
▶ QUINAS:  {total_quinas} cartões ➔ {(total_quinas / total_bilhetes) * 100:.4f}%  (Acaso Teórico: 0.0000%)

➔ GRUPO DE ELITE BASE DO GRAFO (15 Dezenas Analisadas):
   {', '.join(f'{n:02d}' for n in dezenas_master_15)}

➔ PALPITE SUPREMO EMITIDO (SEU 1 VOLANTE REAL):
   {palpite_definitivo} | Índice Real: {jogo_para_indice(palpite_definitivo):,} | Soma: {sum(palpite_definitivo)}
=====================================================================================\n\n"""

print(relatorio_txt)

with open("historico_testes_quina.txt", "a", encoding="utf-8") as f:
    f.write(relatorio_txt)

print("-> Sucesso operacional! O volante de alvo único foi isolado e guardado.")