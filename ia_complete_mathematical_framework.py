import numpy as np
import pandas as pd
import random
import os
from math import comb, log, exp
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO FRAMEWORK MATEMÁTICO COM MOTOR DE ACURÁCIA (BACKTESTING)")
print("=" * 80)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()
total_sorteios = len(df_cronologico)

# CORRIGIDO: Variável global declarada no escopo principal para mitigar o NameError
matriz_vetores_6d = df_cronologico[colunas_bolas].values.astype(np.float32)

# Divisão de Dados para Prova de Acurácia (80% Histórico Base, 20% Bloco Oculto de Teste)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 1. ENGINE MATEMÁTICA REUTILIZÁVEL (FOTO DO PASSADO)
# =============================================================
def calcular_subespaco_e_limites_historicos(df_historico_ponto):
    """
    Calcula as retas, polinômios, logaritmos e limites de integração
    usando estritamente os dados disponíveis até aquele ponto do tempo.
    """
    sorteios_locais = len(df_historico_ponto)
    matriz_v_6d = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    vetor_mu_local = np.mean(matriz_v_6d, axis=0)
    
    # 1.1 Regressão Vetorial Local (Retas de Tendência)
    eixo_t = np.arange(sorteios_locais)
    proj_retas = []
    for b_idx in range(6):
        coef = np.polyfit(eixo_t, matriz_v_6d[:, b_idx], 1)
        proj_retas.append(coef)
        
    # 1.2 Limites de Integração da Casca do Hipercubo
    dist_euclidianas = np.sqrt(np.sum((matriz_v_6d - vetor_mu_local) ** 2, axis=1))
    media_d = np.mean(dist_euclidianas)
    desvio_d = np.std(dist_euclidianas)
    lim_min = media_d - (0.6 * desvio_d)
    lim_max = media_d + (1.2 * desvio_d)
    
    return vetor_mu_local, proj_retas, lim_min, lim_max

def gerar_universo_elite_polinomial(df_historico_ponto):
    """ Monta o polinômio não linear com transformação logarítmica de Bayes """
    sorteios_locais = len(df_historico_ponto)
    
    # Recalcula a matriz de transições para o ponto do passado
    current_ausencia = {d: 0 for d in range(1, 61)}
    todas_dezenas_locais = []
    for index, row in df_historico_ponto.iterrows():
        sorteados = set([row[col] for col in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: current_ausencia[d] = 0
            else: current_ausencia[d] += 1
            if d in sorteados: todas_dezenas_locais.append(d)
            
    frequencia_dezenas = Counter(todas_dezenas_locais)
    
    scores_polinomio = []
    for d in range(1, 61):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - (sorteios_locais / 10)
        jejum = current_ausencia[d]
        
        p_bayes = max(0.0001, (freq / (sorteios_locais + 1e-5)) * (1 / (1 + jejum)))
        log_bayes = log(p_bayes)
        
        # Polinômio Composito
        score = (0.5 * log_bayes) + (0.3 * (freq / (sorteios_locais + 1e-5))) + (0.2 * (desvio ** 2) / (sorteios_locais + 1e-5))
        scores_polinomio.append((d, score))
        
    scores_ordenados = sorted(scores_polinomio, key=lambda x: x[1], reverse=True)
    return [item[0] for item in scores_ordenados[:28]]

def filtrar_cinco_jogos_framework(universo_elite, v_mu, lim_min, lim_max, conjunto_senas_hist):
    """ O Crivo do Raciocínio Lógico Booleano para aprovação de jogos """
    jogos_filtrados = []
    iteracoes = 0
    while len(jogos_filtrados) < 5 and iteracoes < 4000:
        iteracoes += 1
        palpite = sorted(random.sample(universo_elite, 6))
        vetor_np = np.array(palpite, dtype=np.float32)
        
        # Condições Booleanas
        dist_p = np.sqrt(np.sum((vetor_np - v_mu) ** 2))
        c_casca = (lim_min <= dist_p <= lim_max)
        
        c_derivada = (np.max(np.diff(vetor_np)) <= 24 and np.min(np.diff(vetor_np)) >= 1)
        
        c_paridade = (sum(1 for n in palpite if n % 2 == 0) in [2, 3, 4])
        
        c_soma = (120 <= sum(palpite) <= 200)
        
        c_conjuntos = True
        set_palpite = set(palpite)
        for sena_h in conjunto_senas_hist[-150:]:
            if len(set_palpite & sena_h) > 4:
                c_conjuntos = False
                break
                
        if c_casca and c_derivada and c_paridade and c_soma and c_conjuntos:
            if jogo_e_valido_estritamente(palpite):
                if palpite not in jogos_filtrados:
                    jogos_filtrados.append(palpite)
    return jogos_filtrados


# =============================================================
# 2. PROVA DE ACURÁCIA (FASES DE BACKTESTING)
# =============================================================
print(f"\n[FASE 2] Iniciando cálculo de Acurácia Real em {len(df_teste_validacao)} concursos ocultos...")

total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes_testados = 0
conjunto_global_senas = [set(v) for v in matriz_vetores_6d]

for i in range(len(df_teste_validacao)):
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    # Garante corte limpo no tempo para não vazar o futuro para o modelo
    df_historico_ponto = df_cronologico[df_cronologico['Concurso'] < concurso_alvo['Concurso']]
    conjunto_senas_hist_ponto = conjunto_global_senas[:len(df_historico_ponto)]
    
    # Roda o pipeline matemático para o ponto do passado
    v_mu_ponto, proj_retas_ponto, lim_min_ponto, lim_max_ponto = calcular_subespaco_e_limites_historicos(df_historico_ponto)
    elite_ponto = gerar_universo_elite_polinomial(df_historico_ponto)
    
    jogos_gerados_ponto = filtrar_cinco_jogos_framework(elite_ponto, v_mu_ponto, lim_min_ponto, lim_max_ponto, conjunto_senas_hist_ponto)
    
    for jogo in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(jogo) & gabarito_real)
        if acertos == 6: total_senas += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 3: total_ternos += 1

# Cálculo exato das taxas percentuais de acerto
taxa_ternos = (total_ternos / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quadras = (total_quadras / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quinas = (total_quinas / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA MULTIDISCIPLINAR (BACKTESTING COMPLETO):")
print(f"▶ Total de bilhetes válidos construídos matricialmente: {total_bilhetes_testados}")
print("-" * 85)
print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE REAL):")
print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%  (Esperado acaso: 0.4311%)")
print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%  (Esperado acaso: 0.0428%)")
print(f"▶ QUINAS:  {total_quinas} acertos ➔ {taxa_quinas:.4f}%  (Esperado acaso: 0.0006%)")
print(f"▶ SENAS:   {total_senas} acertos")
print("-" * 85)


# =============================================================
# 3. GERAÇÃO DO PALPITE DO PRÓXIMO CONCURSO ATUAL
# =============================================================
print("\n[FASE 3] Rodando Projeção Geométrica para o PRÓXIMO concurso...")
v_mu_atual, proj_retas_atual, lim_min_atual, lim_max_atual = calcular_subespaco_e_limites_historicos(df_cronologico)
elite_atual = gerar_universo_elite_polinomial(df_cronologico)

# CORRIGIDO: Vinculada a variável correta para o laço de projeção temporal futura
vetor_projetado_futuro = np.array([poly[0] * total_sorteios + poly[1] for poly in proj_retas_atual], dtype=np.float32).flatten()
jogos_definitivos = filtrar_cinco_jogos_framework(elite_atual, v_mu_atual, lim_min_atual, lim_max_atual, conjunto_global_senas)

print("\n" + "=" * 85)
print("      JOGOS DEFINITIVOS EMITIDOS VIA ECOSSISTEMA MATEMÁTICO INTEGRADO")
print("=" * 85)
for i, jogo in enumerate(jogos_definitivos, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
    
    dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Distância 6D: {dist_final:.2f} | Soma: {sum(jogo)}")
print("=" * 85)