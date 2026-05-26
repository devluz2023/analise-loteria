import numpy as np
import pandas as pd
import random
import os
import requests
import io
import itertools
from math import log
from collections import Counter

# Abafa logs de aviso do sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO MOTOR MATEMÁTICO AVANÇADO - MODALIDADE: LOTOFÁCIL")
print("=" * 80)

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
print(f"-> Sucesso! {total_sorteios} concursos carregados.")

# =============================================================
# 2. ENGENHARIA DE DADOS: GERAÇÃO DA MATRIZ DE ÍNDICES GLOBAL
# =============================================================
print("Gerando a Matriz de Índices Combinatórios Completa (3.268.760 combinações)...")
# Cria o espaço amostral absoluto de forma matricial contínua na memória
todas_combinacoes_lotofacil = list(itertools.combinations(range(1, 26), 15))
matriz_indices_global = np.array(todas_combinacoes_lotofacil, dtype=np.uint8)
total_combinacoes_possiveis = len(matriz_indices_global)
print(f"-> Espaço amostral indexado com sucesso. Total: {total_combinacoes_possiveis:,} partições.")

# Mapeia o índice correspondente de cada concurso real do histórico para a Série Temporal
print("Mapeando os índices dos concursos passados...")
indices_concursos_historicos = []
# Transforma a matriz global num set de tuplas para busca veloz do índice
mapa_busca_indices = {tuple(combo): idx for idx, combo in enumerate(todas_combinacoes_lotofacil)}

for index, row in df_cronologico.iterrows():
    jogo_real = tuple(sorted([int(row[col]) for col in colunas_bolas]))
    indices_concursos_historicos.append(mapa_busca_indices.get(jogo_real, 0))

df_cronologico['Indice_Combinacao'] = indices_concursos_historicos


# =============================================================
# 3. CONSTRUÇÃO DA MATRIZ HISTÓRICA DE AUSÊNCIA (JEJUM)
# =============================================================
print("Construindo a Matriz Dinâmica de Ausência para as 25 dezenas...")
historico_jejuns = []
atraso_dinamico = {d: 0 for d in range(1, 26)}

for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[col]) for col in colunas_bolas])
    historico_jejuns.append([atraso_dinamico[d] for d in range(1, 26)])
    for d in range(1, 26):
        if d in sorteados: atraso_dinamico[d] = 0
        else: atraso_dinamico[d] += 1

matriz_jejuns_global = np.array(historico_jejuns, dtype=np.int32)

# Divisão de dados para validação da Acurácia Real (80% Base, 20% Bloco de Teste)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 4. MOTOR MATEMÁTICO INTEGRADO (FOTO DO PASSADO)
# =============================================================
def calcular_ambiente_lotofacil(df_historico_ponto, limite_index_ponto):
    """ Calcula os parâmetros geométricos e as tabelas de frequência de Bayes """
    sorteios_locais = len(df_historico_ponto)
    matriz_v = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    
    vetor_mu_local = np.mean(matriz_v, axis=0)
    dist_euclidianas = np.sqrt(np.sum((matriz_v - vetor_mu_local) ** 2, axis=1))
    media_d = np.mean(dist_euclidianas)
    desvio_d = np.std(dist_euclidianas)
    lim_min = media_d - (0.8 * desvio_d)
    lim_max = media_d + (1.2 * desvio_d)
    
    matriz_jejuns_ponto = matriz_jejuns_global[:limite_index_ponto]
    contador_jejuns = Counter(matriz_jejuns_ponto.flatten())
    total_estados_jejum = sum(contador_jejuns.values())
    prob_prior_jejum = {jejum: (qtd / total_estados_jejum) for jejum, qtd in contador_jejuns.items()}
    
    jejum_atual_no_ponto = matriz_jejuns_global[limite_index_ponto - 1]
    todas_dezenas_locais = matriz_v.flatten().astype(np.int32)
    frequencia_dezenas = Counter(todas_dezenas_locais)
    media_frequencia_esperada = sorteios_locais * 15 / 25
    
    score_dezenas_local = {}
    for d in range(1, 26):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - media_frequencia_esperada
        jejum_d = jejum_atual_no_ponto[d - 1]
        
        p_bayes = prob_prior_jejum.get(jejum_d + 1, 0.0001)
        log_bayes = log(p_bayes)
        
        score_genes_local = (0.6 * log_bayes) + (0.2 * (freq / (sorteios_locais + 1e-5))) + (0.2 * (desvio ** 2) / (sorteios_locais + 1e-5))
        score_dezenas_local[d] = score_genes_local
        
    return vetor_mu_local, lim_min, lim_max, score_dezenas_local


def prever_proximo_indice_temporal(serie_indices, window_size=12):
    """ Usa Média Móvel Ponderada Linear na série cronológica de índices """
    janela_recente = serie_indices[-window_size:]
    pesos = np.arange(1, window_size + 1)
    indice_projetado = np.sum(janela_recente * pesos) / np.sum(pesos)
    erro_residual = np.std(janela_recente)
    return int(indice_projetado), int(erro_residual)


def filtrar_jogos_por_busca_matriz(universo_elite, v_mu, lim_min, lim_max, indice_target, desvio_target):
    """ 
    [OTIMIZADO]: Em vez de gerar jogos aleatórios, o modelo varre uma janela 
    da série temporal de índices direto da Matriz de Índices Pré-calculada.
    """
    jogos_filtrados = []
    
    # Define os limites de escaneamento na série temporal de índices
    idx_inicial = max(0, indice_target - desvio_target)
    idx_final = min(total_combinacoes_possiveis - 1, indice_target + desvio_target)
    
    # Cria uma amostragem de índices dentro do intervalo de confiança
    indices_candidatos = random.sample(range(idx_inicial, idx_final + 1), min(5000, idx_final - idx_inicial + 1))
    
    for idx in indices_candidatos:
        # Busca instantânea O(1) na matriz precompilada
        palpite_numpy = matriz_indices_global[idx]
        palpite = [int(n) for n in palpite_numpy]
        
        # Filtro de pertinência: Garante que as dezenas pertençam ao Universo de Elite
        if sum(1 for d in palpite if d in universo_elite) < 12:
            continue
            
        vetor_np = palpite_numpy.astype(np.float32)
        
        # Crivos Geométricos e Físicos
        dist_p = np.sqrt(np.sum((vetor_np - v_mu) ** 2))
        if not (lim_min <= dist_p <= lim_max): continue
        if sum(1 for n in palpite if n % 2 == 0) not in [7, 8]: continue
        if not (175 <= sum(palpite) <= 215): continue
            
        if palpite not in jogos_filtrados:
            jogos_filtrados.append(palpite)
            if len(jogos_filtrados) == 5: break
            
    # Fallback se a janela estiver muito restrita
    while len(jogos_filtrados) < 5:
        backup = sorted(random.sample(universo_elite, 15))
        if backup not in jogos_filtrados:
            jogos_filtrados.append(backup)
            
    return jogos_filtrados


# =============================================================
# 5. PROVA DE ACURÁCIA HISTÓRICA (BACKTESTING ULTRA VELOZ)
# =============================================================
print(f"\n[FASE 3] Executando Backtesting com Matriz de Índices Ativa em {len(df_teste_validacao)} concursos...")

total_11, total_12, total_13, total_14, total_15 = 0, 0, 0, 0, 0
total_bilhetes_testados = 0

for i in range(len(df_teste_validacao)):
    idx_global = split_idx + i
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    df_historico_ponto = df_cronologico.iloc[:idx_global].copy()
    serie_indices_ponto = df_historico_ponto['Indice_Combinacao'].values
    
    v_mu_p, l_min_p, l_max_p, dezenas_p = calcular_ambiente_lotofacil(df_historico_ponto, idx_global)
    elite_p = [item[0] for item in sorted(dezenas_p.items(), key=lambda x: x[1], reverse=True)[:21]]
    
    # Projeta a tendência da série temporal de índices
    idx_target_p, desvio_p = prever_proximo_indice_temporal(serie_indices_ponto)
    
    # Extrai os jogos direto da matriz pré-compilada
    jogos_gerados_ponto = filtrar_jogos_por_busca_matriz(elite_p, v_mu_p, l_min_p, l_max_p, idx_target_p, desvio_p)
    
    for jogo in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(jogo) & gabarito_real)
        if acertos == 15: total_15 += 1
        elif acertos == 14: total_14 += 1
        elif acertos == 13: total_13 += 1
        elif acertos == 12: total_12 += 1
        elif acertos == 11: total_11 += 1

# Conversão em taxas de acurácia
t_11 = (total_11 / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
t_12 = (total_12 / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
t_13 = (total_13 / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA INTEGRADO - PROVA REAL DA LOTOFÁCIL:")
print(f"▶ Total de bilhetes gerados no teste retrospectivo: {total_bilhetes_testados}")
print("-" * 85)
print("TAXA DE CONVERSÃO HISTÓRICA DO FRAMEWORK (HIT RATE DE PREMIAÇÃO COLETIVA):")
print(f"▶ 11 ACERTOS: {total_11} cartões ➔ {t_11:.4f}%  (Esperado acaso: 9.09%)")
print(f"▶ 12 ACERTOS: {total_12} cartões ➔ {t_12:.4f}%  (Esperado acaso: 1.66%)")
print(f"▶ 13 ACERTOS: {total_13} cartões ➔ {t_13:.4f}%  (Esperado acaso: 0.14%)")
print(f"▶ 14 ACERTOS: {total_14} cartões ➔ {(total_14/total_bilhetes_testados)*100:.4f}%")
print(f"▶ 15 ACERTOS: {total_15} cartões")
print("-" * 85)


# =============================================================
# 6. GERAÇÃO DOS JOGOS PARA O PRÓXIMO CONCURSO ATUAL
# =============================================================
print("\n[FASE 4] Calculando geometria de casca e série de índices para o PRÓXIMO concurso...")
v_mu_atual, l_min_atual, l_max_atual, dezenas_atual = calcular_ambiente_lotofacil(df_cronologico, total_sorteios)
serie_indices_completa = df_cronologico['Indice_Combinacao'].values

proximo_idx_target, desvio_atual = prever_proximo_indice_temporal(serie_indices_completa)
elite_atual = [item[0] for item in sorted(dezenas_atual.items(), key=lambda x: x[1], reverse=True)[:21]]

print(f"➔ PREVISÃO CRONOLÓGICA DE ÍNDICE: {proximo_idx_target:,} (Janela de Varredura: ±{desvio_atual:,})")

jogos_definitivos = filtrar_jogos_por_busca_matriz(elite_atual, v_mu_atual, l_min_atual, l_max_atual, proximo_idx_target, desvio_atual)

print("\n" + "=" * 85)
print("             JOGOS DA LOTOFÁCIL EMITIDOS VIA ECOSSISTEMA GEOMÉTRICO")
print("=" * 85)
for i, jogo in enumerate(jogos_definitivos, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
    
    # Calcula o índice real exato do palpite final usando a tabela hash de busca
    idx_real_palpite = mapa_busca_indices.get(tuple(jogo), 0)
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice na Matriz: {idx_real_palpite:,} | Distância 15D: {dist_final:.2f} | Soma: {sum(jogo)}")
print("=" * 85)