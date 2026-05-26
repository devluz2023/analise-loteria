import numpy as np
import pandas as pd
import random
import os
from math import comb
from collections import Counter
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO ECOSSISTEMA COM MATRIZ DE LOGICA FUZZY E ACURÁCIA")
print("=" * 80)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()
total_sorteios = len(df_cronologico)

# Separação estrita para validação da Acurácia (80% Base Histórica, 20% Validação Cega)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 1. FUNÇÕES DE PERTINÊNCIA FUZZY (MEMBERSHIP FUNCTIONS)
# =============================================================
def pertinencia_triangular(x, a, b, c):
    """ Calcula o grau fuzzy [0, 1] em uma curva triangular """
    if x <= a or x >= c: return 0.0
    if a < x <= b: return (x - a) / (b - a)
    if b < x < c: return (c - x) / (c - b)
    return 0.0

def pertinencia_trapezoidal_direita(x, a, b):
    """ Curva para conjuntos abertos à direita (ex: Muito Atrasada / Infinito) """
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a)


# =============================================================
# 2. MOTOR CORE FUZZY: INFERÊNCIA E DEFUZZIFICAÇÃO
# =============================================================
def calcular_scores_fuzzy_historico(df_historico_ponto):
    """
    Fuzzifica o atraso e a frequência de cada dezena, aplica as regras
    e extrai o score resolvido (Defuzzificado) usando apenas o passado.
    """
    sorteios_locais = len(df_historico_ponto)
    
    # Mapeia os atrasos no ponto do tempo
    current_ausencia = {d: 0 for d in range(1, 61)}
    todas_dezenas_locais = []
    for index, row in df_historico_ponto.iterrows():
        sorteados = set([row[col] for col in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: current_ausencia[d] = 0
            else: current_ausencia[d] += 1
            if d in sorteados: todas_dezenas_locais.append(d)
            
    frequencia_dezenas = Counter(todas_dezenas_locais)
    media_frequencia_esperada = sorteios_locais / 10
    
    scores_fuzzy_dezenas = {}
    
    for d in range(1, 61):
        jejum = current_ausencia[d]
        freq = frequencia_dezenas.get(d, 0)
        
        # 2.1 FUZZIFICAÇÃO (Mapeamento dos graus de pertinência)
        # Variável Atraso (Jejum)
        uf_jejum_recente = pertinencia_triangular(jejum, -1, 2, 6)
        uf_jejum_medio = pertinencia_triangular(jejum, 4, 10, 18)
        uf_jejum_longo = pertinencia_trapezoidal_direita(jejum, 14, 28)
        
        # Variável Frequência
        uf_freq_fraca = pertinencia_triangular(freq, -1, media_frequencia_esperada * 0.8, media_frequencia_esperada)
        uf_freq_normal = pertinencia_triangular(freq, media_frequencia_esperada * 0.85, media_frequencia_esperada, media_frequencia_esperada * 1.15)
        uf_freq_forte = pertinencia_trapezoidal_direita(freq, media_frequencia_esperada * 1.1, media_frequencia_esperada * 1.3)
        
        # 2.2 MÁQUINA DE INFERÊNCIA FUZZY (Aplicação das Regras Lógicas Nebulosas)
        # Usamos o operador MIN para a conjunção 'E' (Lógica de Mamdani)
        regra1_atracao_maxima = min(uf_jejum_longo, uf_freq_forte)   # Muito atrasada E historicamente forte
        regra2_atracao_alta = min(uf_jejum_medio, uf_freq_forte)     # Atraso médio E historicamente forte
        regra3_atracao_media = min(uf_jejum_recente, uf_freq_normal) # Recente E frequência normal
        regra4_atracao_baixa = min(uf_jejum_recente, uf_freq_fraca)  # Recente E frequência fraca
        
        # 2.3 DEFUZZIFICAÇÃO (Centro de Gravidade simplificado / Média Ponderada)
        # Atribuímos pesos numéricos nítidos para os conjuntos de saída (Singletone Fuzzy)
        # Máxima = 1.0 | Alta = 0.75 | Média = 0.5 | Baixa = 0.1
        numerador = (regra1_atracao_maxima * 1.0) + (regra2_atracao_alta * 0.75) + (regra3_atracao_media * 0.5) + (regra4_atracao_baixa * 0.1)
        denominador = regra1_atracao_maxima + regra2_atracao_alta + regra3_atracao_media + regra4_atracao_baixa
        
        if denominador > 0:
            score_final = numerador / denominador
        else:
            score_final = 0.3 # Score base neutro de segurança
            
        scores_fuzzy_dezenas[d] = score_final
        
    return scores_fuzzy_dezenas

def amostrar_cinco_jogos_fuzzy(scores_fuzzy_ponto):
    """ Filtra as melhores dezenas fuzzy e monta os 5 cartões estáveis """
    dezenas_ordenadas = sorted(scores_fuzzy_ponto.items(), key=lambda x: x[1], reverse=True)
    universo_elite_fuzzy = [item[0] for item in dezenas_ordenadas[:26]]
    
    jogos_filtrados = []
    tentativas = 0
    while len(jogos_filtrados) < 5 and tentativas < 10000:
        tentativas += 1
        palpite = sorted(random.sample(universo_elite_fuzzy, 6))
        
        # Restrições de Distribuição Físicas do seu Framework
        if sum(1 for n in palpite if n % 2 == 0) not in [2, 3, 4]: continue
        if not (120 <= sum(palpite) <= 200): continue
            
        if jogo_e_valido_estritamente(palpite):
            if palpite not in jogos_filtrados:
                jogos_filtrados.append(palpite)
                
    return jogos_filtrados


# =============================================================
# 3. PROVA DE ACURÁCIA DO SISTEMA FUZZY (BACKTESTING)
# =============================================================
print(f"\n[FASE 2] Executando Backtesting Fuzzy em {len(df_teste_validacao)} concursos ocultos...")

total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes_testados = 0

for i in range(len(df_teste_validacao)):
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    # Separação no tempo para congelar o passado do concurso testado
    df_historico_ponto = df_cronologico[df_cronologico['Concurso'] < concurso_alvo['Concurso']]
    
    # Roda as regras lógicas fuzzy daquela época do passado
    scores_fuzzy_ponto = calcular_scores_fuzzy_historico(df_historico_ponto)
    jogos_fuzzy_ponto = amostrar_cinco_jogos_fuzzy(scores_fuzzy_ponto)
    
    for jogo in jogos_fuzzy_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(jogo) & gabarito_real)
        if acertos == 6: total_senas += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 3: total_ternos += 1

# Taxas percentuais de acurácia convertidas
taxa_ternos = (total_ternos / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
taxa_quadras = (total_quadras / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA DO INFERIDOR LOGICO FUZZY:")
print(f"▶ Total de cartões validados no teste cego: {total_bilhetes_testados}")
print("-" * 85)
print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE FUZZY):")
print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
print(f"▶ QUINAS:  {total_quinas} acertos")
print("-" * 85)


# =============================================================
# 4. PALPITE DEFINITIVO PARA O PRÓXIMO CONCURSO REAL
# =============================================================
print("\n[FASE 3] Calculando graus de verdade fuzzy para o PRÓXIMO concurso real...")
scores_fuzzy_atual = calcular_scores_fuzzy_historico(df_cronologico)
jogos_finais = amostrar_cinco_jogos_fuzzy(scores_fuzzy_atual)

print("\n" + "=" * 85)
print("      JOGOS DEFINITIVOS EMITIDOS VIA SISTEMA DE INFERÊNCIA LÓGICA FUZZY")
print("=" * 85)
for i, jogo in enumerate(jogos_finais, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
    
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Soma: {sum(jogo)}")
print("=" * 85)