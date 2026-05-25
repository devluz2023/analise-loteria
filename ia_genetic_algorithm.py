import numpy as np
import pandas as pd
import random
import optuna
import os
from math import log
from collections import Counter
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO ECOSSISTEMA EVOLUTIVO (AG) COM MOTOR DE ACURÁCIA")
print("=" * 80)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()
total_sorteios = len(df_cronologico)

# Divisão de Dados para Prova de Acurácia (80% Histórico Base, 20% Bloco Oculto de Teste)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 1. MOTOR DE EVOLUÇÃO E NOTA DE APTIDÃO (FOTO DO PASSADO)
# =============================================================
def extrair_parametros_e_genes_historicos(df_historico_ponto):
    """ Calcula a casca do hipercubo e pesos polinomiais baseados no passado """
    sorteios_locais = len(df_historico_ponto)
    matriz_v_6d = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    vetor_mu_local = np.mean(matriz_v_6d, axis=0)
    
    dist_euclidianas = np.sqrt(np.sum((matriz_v_6d - vetor_mu_local) ** 2, axis=1))
    media_d, desvio_d = np.mean(dist_euclidianas), np.std(dist_euclidianas)
    lim_min = media_d - (0.6 * desvio_d)
    lim_max = media_d + (1.2 * desvio_d)
    
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
    
    score_genes_local = {}
    for d in range(1, 61):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - (sorteios_locais / 10)
        jejum = current_ausencia[d]
        
        p_bayes = max(0.0001, (freq / (sorteios_locais + 1e-5)) * (1 / (1 + jejum)))
        log_bayes = log(p_bayes)
        
        score_genes_local[d] = (0.5 * log_bayes) + (0.3 * (freq / (sorteios_locais + 1e-5))) + (0.2 * (desvio ** 2) / (sorteios_locais + 1e-5))
        
    return vetor_mu_local, lim_min, lim_max, score_genes_local

def calcular_fitness_jogo(jogo, score_genes, v_mu, l_min, l_max):
    """ Função de Aptidão: pontua o nível de adaptação biológica do bilhete """
    vetor_np = np.array(jogo, dtype=np.float32)
    score_fitness = sum(score_genes.get(d, -5.0) for d in jogo)
    
    dist_p = np.sqrt(np.sum((vetor_np - v_mu) ** 2))
    if not (l_min <= dist_p <= l_max):
        score_fitness -= 50.0 # Fora da casca externa
        
    if not (120 <= sum(jogo) <= 200):
        score_fitness -= 30.0 # Soma fora da curva normal
        
    if sum(1 for n in jogo if n % 2 == 0) not in [2, 3, 4]:
        score_fitness -= 20.0 # Paridade desequilibrada
        
    if not jogo_e_valido_estritamente(jogo):
        score_fitness -= 100.0 # Reprovado nos filtros de lógica do Excel
        
    return score_fitness

def crossover_uniforme(pai1, pai2, score_genes):
    filho_genes = set()
    pool_genes = list(set(pai1) | set(pai2))
    pool_genes = sorted(pool_genes, key=lambda d: score_genes.get(d, -5), reverse=True)
    
    for gene in pool_genes:
        if len(filho_genes) < 6:
            if random.random() < 0.7:
                filho_genes.add(gene)
    while len(filho_genes) < 6:
        filho_genes.add(random.randint(1, 60))
    return sorted(list(filho_genes))

def mutacao_aleatoria(jogo, taxa_mutacao=0.08):
    if random.random() < taxa_mutacao:
        idx_mutar = random.randint(0, 5)
        while True:
            novo_gene = random.randint(1, 60)
            if novo_gene not in jogo:
                jogo[idx_mutar] = novo_gene
                break
    return sorted(jogo)

def rodar_ciclo_evolutivo(score_genes, v_mu, l_min, l_max, tam_pop=80, tx_mut=0.08, geracoes=30):
    """ Executa a seleção natural para convergir nos 5 melhores jogos possíveis """
    populacao = []
    while len(populacao) < tam_pop:
        cromossomo = sorted(random.sample(range(1, 61), 6))
        if cromossomo not in populacao:
            populacao.append(cromossomo)
            
    for _ in range(geracoes):
        pop_avaliada = [(ind, calcular_fitness_jogo(ind, score_genes, v_mu, l_min, l_max)) for ind in populacao]
        pop_avaliada = sorted(pop_avaliada, key=lambda x: x[1], reverse=True)
        
        corte_elite = max(2, int(tam_pop * 0.2))
        nova_pop = [item[0] for item in pop_avaliada[:corte_elite]]
        
        while len(nova_pop) < tam_pop:
            p1 = random.choice(nova_pop[:corte_elite])
            p2 = random.choice(nova_pop[:corte_elite])
            filho = crossover_uniforme(p1, p2, score_genes)
            filho = mutacao_aleatoria(filho, tx_mut)
            nova_pop.append(filho)
        populacao = nova_pop
        
    # Extrai o Top 5 final altamente adaptado
    pop_final = sorted([(ind, calcular_fitness_jogo(ind, score_genes, v_mu, l_min, l_max)) for ind in populacao], key=lambda x: x[1], reverse=True)
    retorno = []
    for item in pop_final:
        if item[0] not in retorno:
            retorno.append(item[0])
            if len(retorno) == 5: break
    return retorno


# =============================================================
# 2. PROVA DE ACURÁCIA REAL (BACKTESTING)
# =============================================================
print(f"\n[FASE 2] Calculando Acurácia Real em {len(df_teste_validacao)} concursos com evolução genética...")

total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes_testados = 0

for i in range(len(df_teste_validacao)):
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    # Isola o histórico para evitar vazamento de dados do futuro
    df_historico_ponto = df_cronologico[df_cronologico['Concurso'] < concurso_alvo['Concurso']]
    
    # Extrai as forças daquela época do passado
    v_mu_ponto, l_min_ponto, l_max_ponto, genes_ponto = extrair_parametros_e_genes_historicos(df_historico_ponto)
    
    # Roda a evolução natural simulando o passado (Gera os 5 mais adaptados da época)
    jogos_gerados_ponto = rodar_ciclo_evolutivo(genes_ponto, v_mu_ponto, l_min_ponto, l_max_ponto, tam_pop=80, tx_mut=0.08, geracoes=25)
    
    for jogo in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(jogo) & gabarito_real)
        if acertos == 6: total_senas += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 3: total_ternos += 1

# Conversão estatística das taxas de acerto reais obtidas pelo AG
taxa_ternos = (total_ternos / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quadras = (total_quadras / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quinas = (total_quinas / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA DO MOTOR GENÉTICO EVOLUTIVO:")
print(f"▶ Total de indivíduos testados na barreira do tempo: {total_bilhetes_testados}")
print("-" * 85)
print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE DO PROCESSO SELETIVO):")
print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%  (Esperado acaso: 0.4311%)")
print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%  (Esperado acaso: 0.0428%)")
print(f"▶ QUINAS:  {total_quinas} acertos ➔ {taxa_quinas:.4f}%  (Esperado acaso: 0.0006%)")
print(f"▶ SENAS:   {total_senas} acertos")
print("-" * 85)


# =============================================================
# 3. GERAÇÃO DO PALPITE DO PRÓXIMO CONCURSO REAL
# =============================================================
print("\n[FASE 3] Executando evolução máxima (50 gerações) para o PRÓXIMO concurso real...")
v_mu_atual, l_min_atual, l_max_atual, genes_atual = extrair_parametros_e_genes_historicos(df_cronologico)

# No palpite oficial, aumentamos a população e as gerações para refinar a busca
jogos_definitivos = rodar_ciclo_evolutivo(genes_atual, v_mu_atual, l_min_atual, l_max_atual, tam_pop=120, tx_mut=0.06, geracoes=50)

print("\n" + "=" * 85)
print("      JOGOS DEFINITIVOS EMITIDOS VIA EVOLUÇÃO DE ALGORITMO GENÉTICO")
print("=" * 85)
for i, jogo in enumerate(jogos_definitivos, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
    
    dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Distância 6D: {dist_final:.2f} | Soma: {sum(jogo)}")
print("=" * 85)