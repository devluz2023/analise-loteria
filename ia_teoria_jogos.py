import numpy as np
import pandas as pd
import random
import os
from math import log
from collections import Counter
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO MOTOR DE TEORIA DOS JOGOS (ESTRATÉGIA ANTAGONISTA COM ACURÁCIA)")
print("=" * 80)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()
total_sorteios = len(df_cronologico)

# Divisão de dados para validação da Acurácia Real (80% Base, 20% Teste)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 1. ENGENHARIA DE MATRIZ DE PAYOFF (TEORIA DOS JOGOS)
# =============================================================
def calcular_matriz_payoff_historica(df_historico_ponto):
    """
    Calcula a utilidade estatística de cada dezena, subtraindo a
    probabilidade de colisão/rateio baseada na psicologia das massas.
    """
    sorteios_locais = len(df_historico_ponto)
    
    current_ausencia = {d: 0 for d in range(1, 61)}
    todas_dezenas_locais = []
    for index, row in df_historico_ponto.iterrows():
        sorteados = set([row[col] for col in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: current_ausencia[d] = 0
            else: current_ausencia[d] += 1
            if d in sorteados: todas_dezenas_locais.append(d)
            
    frequencia_dezenas = Counter(todas_dezenas_locais)
    
    # Vetor de Utilidade do Jogador (Antagonismo à Massa)
    scores_payoff = {}
    
    for d in range(1, 61):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - (sorteios_locais / 10)
        jejum = current_ausencia[d]
        
        # Força Pura da Matemática (Bayes + Histórico)
        p_matematica = max(0.0001, (freq / (sorteios_locais + 1e-5)) * (1 / (1 + jejum)))
        
        # --- MODELAGEM PSICOLÓGICA DA MASSA (PENALIDADE DE COLLISION) ---
        penalidade_massa = 0.0
        if d <= 31:
            penalidade_massa += 0.35 # Altíssima concentração devido a datas (aniversários)
        if d in [7, 13, 21, 33, 44]:
            penalidade_massa += 0.20 # Números de apelo subjetivo / "sorte"
            
        # Função de Utilidade Estratégica de Nash (Maximizar ganho científico, minimizar rateio)
        # O log da probabilidade matemática é balanceado pela esquiva da massa
        scores_payoff[d] = log(p_matematica) - penalidade_massa + (0.1 * (desvio ** 2) / (sorteios_locais + 1e-5))
        
    return scores_payoff

def filtrar_jogos_estrategia_mista(scores_payoff, df_historico_ponto):
    """ Amostra jogos focando na região de equilíbrio de Nash do espaço vetorial """
    dezenas_ordenadas = sorted(scores_payoff.items(), key=lambda x: x[1], reverse=True)
    # Seleciona o subconjunto das 28 dezenas de melhor Payoff Coletivo (Equilíbrio Estratégico)
    universo_equilibrio = [item[0] for item in dezenas_ordenadas[:28]]
    
    # Parâmetros de casca geométrica do passado
    matriz_v = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    v_mu = np.mean(matriz_v, axis=0)
    dist_e = np.sqrt(np.sum((matriz_v - v_mu) ** 2, axis=1))
    lim_min = np.mean(dist_e) - (0.6 * np.std(dist_e))
    lim_max = np.mean(dist_e) + (1.2 * np.std(dist_e))
    
    jogos_aprovados = []
    tentativas = 0
    while len(jogos_aprovados) < 5 and tentativas < 10000:
        tentativas += 1
        palpite = sorted(random.sample(universo_equilibrio, 6))
        vetor_np = np.array(palpite, dtype=np.float32)
        
        # Filtros Físicos e Geométricos Unificados
        dist_p = np.sqrt(np.sum((vetor_np - v_mu) ** 2))
        if not (lim_min <= dist_p <= lim_max): continue
        if not (120 <= sum(palpite) <= 200): continue
        if sum(1 for n in palpite if n % 2 == 0) not in [2, 3, 4]: continue
            
        # Restrição Adicional de Teoria dos Jogos: Evita que mais de 3 números pertençam à zona de datas (<=31)
        # Isso garante blindagem contra rateio massivo
        if sum(1 for n in palpite if n <= 31) > 3:
            continue
            
        if jogo_e_valido_estritamente(palpite):
            if palpite not in jogos_aprovados:
                jogos_aprovados.append(palpite)
                
    return jogos_aprovados


# =============================================================
# 2. PROVA DE ACURÁCIA (BACKTESTING DA MATRIZ DE PAYOFF)
# =============================================================
print(f"\n[FASE 2] Executando Backtesting de Teoria dos Jogos em {len(df_teste_validacao)} concursos...")

total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes_testados = 0

for i in range(len(df_teste_validacao)):
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    df_historico_ponto = df_cronologico[df_cronologico['Concurso'] < concurso_alvo['Concurso']]
    
    # Roda o equilíbrio com fotos do passado
    payoff_ponto = calcular_matriz_payoff_historica(df_historico_ponto)
    jogos_gerados_ponto = filtrar_jogos_estrategia_mista(payoff_ponto, df_historico_ponto)
    
    for jogo in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(jogo) & gabarito_real)
        if acertos == 6: total_senas += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 3: total_ternos += 1

taxa_ternos = (total_ternos / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quadras = (total_quadras / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA ESTRATÉGICA (NASH EQUILIBRIUM):")
print(f"▶ Total de bilhetes submetidos à simulação competitiva: {total_bilhetes_testados}")
print("-" * 85)
print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE):")
print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
print(f"▶ QUINAS:  {total_quinas} acertos")
print("-" * 85)


# =============================================================
# 3. GERAÇÃO DOS JOGOS PARA O PRÓXIMO CONCURSO REAL
# =============================================================
print("\n[FASE 3] Calculando Equilíbrio de Nash e utilidades para o PRÓXIMO concurso real...")
payoff_atual = calcular_matriz_payoff_historica(df_cronologico)
jogos_finais = filtrar_jogos_estrategia_mista(payoff_atual, df_cronologico)

print("\n" + "=" * 85)
print("      JOGOS DEFINITIVOS: EQUILÍBRIO DE NASH (MÁXIMO PAYOFF ANTI-RATEIO)")
print("=" * 85)
for i, jogo in enumerate(jogos_finais, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
    
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Soma: {sum(jogo)}")
print("=" * 85)