import numpy as np
import pandas as pd
import random
import os
import requests
import io
import itertools
from math import log, exp, pi
from collections import Counter

# Abafa logs de aviso do sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  RECALIBRANDO CIRCUITO QUÂNTICO - FLEXIBILIZAÇÃO NÃO LINEAR DE ÍNDICES")
print("=" * 85)

# =============================================================
# 1. DOWNLOAD E PREPARAÇÃO DOS DADOS DA LOTOFÁCIL
# =============================================================
print("Baixando dados oficiais da Lotofácil diretamente da Caixa...")
url_lotofacil = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Lotofacil"

response = requests.get(url_lotofacil, verify=False)
response.raise_for_status()

df_lotofacil = pd.read_excel(io.BytesIO(response.content))
colunas_bolas = [f'Bola{i}' for i in range(1, 16)]

df_cronologico = df_lotofacil.sort_values(by='Concurso').copy().reset_index(drop=True)
total_sorteios = len(df_cronologico)

# Precompilação da Matriz de Índices Global para aceleração quântica
todas_combinacoes = list(itertools.combinations(range(1, 26), 15))
matriz_indices_global = np.array(todas_combinacoes, dtype=np.uint8)
total_combinacoes_possiveis = len(matriz_indices_global)

mapa_busca_indices = {tuple(combo): idx for idx, combo in enumerate(todas_combinacoes)}
indices_concursos_historicos = [mapa_busca_indices.get(tuple(sorted([int(row[col]) for col in colunas_bolas])), 0) for _, row in df_cronologico.iterrows()]
df_cronologico['Indice_Combinacao'] = indices_concursos_historicos

# Matriz de Ausência (Jejum)
historico_jejuns = []
atraso_dinamico = {d: 0 for d in range(1, 26)}
for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[col]) for col in colunas_bolas])
    historico_jejuns.append([atraso_dinamico[d] for d in range(1, 26)])
    for d in range(1, 26):
        if d in sorteados: atraso_dinamico[d] = 0
        else: atraso_dinamico[d] += 1
matriz_jejuns_global = np.array(historico_jejuns, dtype=np.int32)

split_idx = int(total_sorteios * 0.8)
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 2. SISTEMA DE FILTRAGEM GEOMÉTRICA E ADJACÊNCIA
# =============================================================
def validar_restricao_sequencial(jogo):
    diferencas = np.diff(jogo)
    sequencia_atual = 1
    max_sequencia = 1
    for diff in diferencas:
        if diff == 1:
            sequencia_atual += 1
            if sequencia_atual > max_sequencia: max_sequencia = sequencia_atual
        else: sequencia_atual = 1
    if max_sequencia >= 4: return False
    if sum(1 for diff in diferencas if diff == 1) > 8: return False
    return True

def validar_restricao_inercia_passada(jogo, ultimo_resultado_real):
    if ultimo_resultado_real is None or len(ultimo_resultado_real) == 0: return True
    total_repetidos = len(set(jogo) & ultimo_resultado_real)
    if total_repetidos < 8 or total_repetidos > 11: return False
    return True


# =============================================================
# 3. [CORREÇÃO] AVALIADOR DE COERÊNCIA DE ÍNDICE (FLEXÍVEL)
# =============================================================
def avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2):
    """
    Em vez de dar return False e banir o cartão, avalia uma taxa de score.
    Permite saltos naturais e só rejeita se o jogo for extremamente absurdo.
    """
    if idx_n1 == 0 or idx_n2 == 0: return True
        
    idx_atual = mapa_busca_indices.get(tuple(jogo), 0)
    
    # Proteção de proximidade imediata: evita jogos idênticos ou colados na vizinhança
    if abs(idx_atual - idx_n1) < 15000: # Reduzido de 60k para 15k para não sufocar o prêmio
        return False
        
    salto_anterior = idx_n1 - idx_n2
    salto_atual_proposto = idx_atual - idx_n1
    
    # Aliviado: só barra se a aceleração for assustadoramente linear (perfeita)
    if abs(salto_atual_proposto - salto_anterior) < 8000: # Reduzido de 40k para 8k
        return False
        
    return True


# =============================================================
# 4. SINTONIZAÇÃO QUÂNTICA E CONSTRUTOR DE COBERTURA
# =============================================================
def extrair_elite_quantica_focalizada(df_historico_ponto, limite_index_ponto):
    sorteios_locais = len(df_historico_ponto)
    matriz_v = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    vetor_mu_local = np.mean(matriz_v, axis=0)
    
    matriz_jejuns_ponto = matriz_jejuns_global[:limite_index_ponto]
    contador_jejuns = Counter(matriz_jejuns_ponto.flatten())
    total_estados = sum(contador_jejuns.values())
    prob_prior_jejum = {jejum: (qtd / total_estados) for jejum, qtd in contador_jejuns.items()}
    
    jejum_atual = matriz_jejuns_global[limite_index_ponto - 1]
    frequencia_dezenas = Counter(matriz_v.flatten().astype(np.int32))
    media_freq = sorteios_locais * 15 / 25
    
    ultimos_dois = [set(matriz_v[-1].astype(np.int32)), set(matriz_v[-2].astype(np.int32))] if sorteios_locais >= 2 else []
    amplitudes_quânticas = np.zeros(25, dtype=np.complex64)
    
    for d in range(1, 26):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - media_freq
        theta = 2.0 * np.arcsin(np.sqrt(prob_prior_jejum.get(jejum_atual[d - 1] + 1, 0.0001)))
        
        fase_base = exp(-0.5 * (desvio / (np.std(list(frequencia_dezenas.values())) + 1e-5))**2) * pi
        if len(ultimos_dois) == 2 and d in ultimos_dois[0] and d in ultimos_dois[1]:
            fase_base *= 0.75 # Suavizado ligeiramente o freio para não expulsar a 15ª bola
                
        amplitudes_quânticas[d - 1] = np.cos(theta / 2.0) + np.exp(1j * fase_base) * np.sin(theta / 2.0)
        
    media_amp = np.mean(amplitudes_quânticas)
    probabilidades_reais = np.abs(2.0 * media_amp - amplitudes_quânticas) ** 2
    probabilidades_reais /= np.sum(probabilidades_reais)
    
    dezenas_ordenadas = sorted([(d, float(probabilidades_reais[d - 1])) for d in range(1, 26)], key=lambda x: x[1], reverse=True)
    return [item[0] for item in dezenas_ordenadas[:19]], vetor_mu_local


def gerar_cobertura_maxima_19(grupo_19, ultimo_resultado_real, idx_n1, idx_n2):
    indices_fechamento = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 12, 13],
        [0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4],
        [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18],
        [0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 18],
        [0, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18]
    ]
    
    blocos_finais = []
    elementos_fora_global = list(set(range(1, 26)) - set(grupo_19))
    
    for posicoes in indices_fechamento:
        cartao = sorted([grupo_19[idx] for idx in posicoes])
        
        if not validar_restricao_sequencial(cartao):
            elementos_fora = list(set(grupo_19) - set(cartao))
            if elementos_fora: cartao[4] = elementos_fora[0]; cartao = sorted(cartao)
                
        # Validação conjunta com o novo crivo dinâmico suavizado
        tentativas = 0
        while (not validar_restricao_inercia_passada(cartao, ultimo_resultado_real) or \
               not avaliar_coerencia_indices_fuzzy(cartao, idx_n1, idx_n2)) and tentativas < 30:
            tentativas += 1
            idx_sub = random.randint(0, 14)
            if elementos_fora_global:
                cartao[idx_sub] = random.choice(elementos_fora_global)
                cartao = sorted(cartao)
                
        paridade = sum(1 for n in cartao if n % 2 == 0)
        if paridade not in [7, 8, 9]:
            fora_do_cartao = list(set(grupo_19) - set(cartao))
            if fora_do_cartao: cartao[-1] = fora_do_cartao[0]; cartao = sorted(cartao)
                
        blocos_finais.append(cartao)
    return blocos_finais


# =============================================================
# 5. PROVA DE ACURÁCIA (BACKTESTING)
# =============================================================
print(f"\n[FASE 2] Executando Backtesting Calibrado em {len(df_teste_validacao)} concursos...")

total_11, total_12, total_13, total_14, total_15 = 0, 0, 0, 0, 0
total_bilhetes_testados = 0
concursos_com_premio_maximo = 0

for i in range(len(df_teste_validacao)):
    idx_global = split_idx + i
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    df_historico_ponto = df_cronologico.iloc[:idx_global].copy()
    ultimo_resultado_passado = set(df_historico_ponto.iloc[-1][colunas_bolas].values.astype(np.int32)) if len(df_historico_ponto) > 0 else None
    
    idx_n1_passado = df_historico_ponto.iloc[-1]['Indice_Combinacao'] if len(df_historico_ponto) > 0 else 0
    idx_n2_passado = df_historico_ponto.iloc[-2]['Indice_Combinacao'] if len(df_historico_ponto) > 1 else 0
    
    elite_19_passado, _ = extrair_elite_quantica_focalizada(df_historico_ponto, idx_global)
    jogos_gerados_ponto = gerar_cobertura_maxima_19(elite_19_passado, ultimo_resultado_passado, idx_n1_passado, idx_n2_passado)
    
    acertou_alta_premiacao = False
    for juego in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(juego) & gabarito_real)
        if acertos >= 14: acertou_alta_premiacao = True
        if acertos == 15: total_15 += 1
        elif acertos == 14: total_14 += 1
        elif acertos == 13: total_13 += 1
        elif acertos == 12: total_12 += 1
        elif acertos == 11: total_11 += 1
        
    if acertou_alta_premiacao: concursos_com_premio_maximo += 1

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA DA COBERTURA QUÂNTICA RECALIBRADA:")
print(f"▶ Total de bilhetes colapsados e conferidos na malha: {total_bilhetes_testados}")
print("-" * 85)
print("BALANÇO EFETIVO DE PREMIAÇÕES (ACURÁCIA CONSOLIDADA):")
print(f"▶ 11 ACERTOS: {total_11} cartões")
print(f"▶ 12 ACERTOS: {total_12} cartões")
print(f"▶ 13 ACERTOS: {total_13} cartões")
print(f"▶ 14 ACERTOS: {total_14} cartões")
print(f"▶ 15 ACERTOS: {total_15} cartões")
print(f"▶ Concursos com Alta Premiação (14 ou 15): {concursos_com_premio_maximo} concursos.")
print("-" * 85)


# =============================================================
# 6. GERAÇÃO DO FECHAMENTO REAL PARA O PRÓXIMO CONCURSO
# =============================================================
print("\n[FASE 3] Executando sintonização e desdobramento para o PRÓXIMO concurso real...")
elite_19_atual, v_mu_atual = extrair_elite_quantica_focalizada(df_cronologico, total_sorteios)

ultimo_concurso_caixa = set(df_cronologico.iloc[-1][colunas_bolas].values.astype(np.int32))
idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

print(f"➔ CONDENSADO QUÂNTICO DE ELITE SELECIONADO (19 Números Rebalanceados):")
print(f"  {', '.join(f'{n:02d}' for n in elite_19_atual)}\n")

jogos_definitivos_cobertura = gerar_cobertura_maxima_19(elite_19_atual, ultimo_concurso_caixa, idx_n1_real, idx_n2_real)

print("=" * 85)
print("       JOGOS DA LOTOFÁCIL EMITIDOS VIA REDE DE COBERTURA QUÂNTICA MAX")
print("=" * 85)
for i, jogo in enumerate(jogos_definitivos_cobertura, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
    idx_real_palpite = mapa_busca_indices.get(tuple(jogo), 0)
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice na Matriz: {idx_real_palpite:,} | Distância 15D: {dist_final:.2f}")
print("=" * 85)