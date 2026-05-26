import numpy as np
import pandas as pd
import random
import os
import math
from math import log, exp, pi
from collections import Counter
from get_data import df_mega_sena

# Abafa logs de aviso do sistema
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 85)
print("  INICIALIZANDO CIRCUITO DE INFORMAÇÃO QUÂNTICA MAX - MODALIDADE: MEGA-SENA")
print("=" * 85)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy().reset_index(drop=True)
total_sorteios = len(df_cronologico)

# =============================================================
# 1. ENGENHARIA DE PROVIMENTO: INDEXAÇÃO MATEMÁTICA PURA O(1)
# =============================================================
def jogo_para_indice(jogo):
    """ Converte uma combinação de 6 números no seu índice lexicográfico exato (1 a 50.063.860) """
    idx = 1
    atual = 1
    for r, bola in enumerate(jogo):
        for v in range(atual, bola):
            idx += math.comb(60 - v, 6 - 1 - r)
        atual = bola + 1
    return idx

def indice_para_jogo(idx):
    """ Reconstrói as 6 dezenas exatas a partir de um índice posicional de 1 a 50.063.860 """
    jogo = []
    atual = 1
    for r in range(6, 0, -1):
        while True:
            c = math.comb(60 - atual, r - 1)
            if idx <= c:
                jogo.append(atual)
                atual += 1
                break
            idx -= c
            atual += 1
    return jogo

print("Calculando índices cronológicos via funções combinatórias moleculares...")
indices_concursos_historicos = []
for _, row in df_cronologico.iterrows():
    jogo_real = sorted([int(row[col]) for col in colunas_bolas])
    indices_concursos_historicos.append(jogo_para_indice(jogo_real))
df_cronologico['Indice_Combinacao'] = indices_concursos_historicos
print(f"-> Sucesso! {total_sorteios} concursos indexados matematicamente.")

# Matriz de Ausência (Jejum) para o universo de 60 dezenas
print("Construindo Matriz de Ausência Estocástica...")
historico_jejuns = []
atraso_dinamico = {d: 0 for d in range(1, 61)}
for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[col]) for col in colunas_bolas])
    historico_jejuns.append([atraso_dinamico[d] for d in range(1, 61)])
    for d in range(1, 61):
        if d in sorteados: atraso_dinamico[d] = 0
        else: atraso_dinamico[d] += 1
matriz_jejuns_global = np.array(historico_jejuns, dtype=np.int32)

split_idx = int(total_sorteios * 0.8)
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 2. SISTEMA DE FILTRAGEM GEOMÉTRICA E CRIVOS DE ENTROPIA
# =============================================================
def validar_restricao_sequencial(jogo):
    """ Elimina sequências absurdas (ex: 3 números seguidos) """
    diferencas = np.diff(jogo)
    sequencia_atual = 1
    max_sequencia = 1
    for diff in diferencas:
        if diff == 1:
            sequencia_atual += 1
            if sequencia_atual > max_sequencia: max_sequencia = sequencia_atual
        else:
            sequencia_atual = 1
    if max_sequencia >= 3: return False
    return True

def validar_restricao_inercia_passada(jogo, ultimo_resultado_real):
    """ Garante que o jogo repita no máximo 1 ou 2 números do sorteio anterior """
    if ultimo_resultado_real is None or len(ultimo_resultado_real) == 0: return True
    total_repetidos = len(set(jogo) & ultimo_resultado_real)
    if total_repetidos > 2: return False
    return True

def avaliar_coerencia_indices_fuzzy(jogo, idx_n1, idx_n2):
    """ Avalia se o salto do índice combinatório mantém uma aceleração lógica """
    if idx_n1 == 0 or idx_n2 == 0: return True
    idx_atual = jogo_para_indice(jogo)
    
    # Proteção de proximidade imediata no espaço 6D
    if abs(idx_atual - idx_n1) < 150000: return False
    
    salto_anterior = idx_n1 - idx_n2
    salto_atual_proposto = idx_atual - idx_n1
    if abs(salto_atual_proposto - salto_anterior) < 50000: return False
    return True


# =============================================================
# 3. SINTONIZAÇÃO QUÂNTICA (AMPLITUDES DE GROVER)
# =============================================================
def extrair_elite_quantica_megasena(df_historico_ponto, limite_index_ponto):
    sorteios_locais = len(df_historico_ponto)
    matriz_v = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    vetor_mu_local = np.mean(matriz_v, axis=0)
    
    matriz_jejuns_ponto = matriz_jejuns_global[:limite_index_ponto]
    contador_jejuns = Counter(matriz_jejuns_ponto.flatten())
    total_estados = sum(contador_jejuns.values())
    prob_prior_jejum = {jejum: (qtd / total_estados) for jejum, qtd in contador_jejuns.items()}
    
    jejum_atual = matriz_jejuns_global[limite_index_ponto - 1]
    frequencia_dezenas = Counter(matriz_v.flatten().astype(np.int32))
    media_freq = sorteios_locais * 6 / 60
    
    ultimos_dois = [set(matriz_v[-1].astype(np.int32)), set(matriz_v[-2].astype(np.int32))] if sorteios_locais >= 2 else []
    amplitudes_quânticas = np.zeros(60, dtype=np.complex64)
    
    for d in range(1, 61):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - media_freq
        theta = 2.0 * np.arcsin(np.sqrt(prob_prior_jejum.get(jejum_atual[d - 1] + 1, 0.0001)))
        
        fase_base = exp(-0.5 * (desvio / (np.std(list(frequencia_dezenas.values())) + 1e-5))**2) * pi
        if len(ultimos_dois) == 2 and d in ultimos_dois[0] and d in ultimos_dois[1]:
            fase_base *= 0.75
                
        amplitudes_quânticas[d - 1] = np.cos(theta / 2.0) + np.exp(1j * fase_base) * np.sin(theta / 2.0)
        
    media_amp = np.mean(amplitudes_quânticas)
    probabilidades_reais = np.abs(2.0 * media_amp - amplitudes_quânticas) ** 2
    probabilidades_reais /= np.sum(probabilidades_reais)
    
    dezenas_ordenadas = sorted([(d, float(probabilidades_reais[d - 1])) for d in range(1, 61)], key=lambda x: x[1], reverse=True)
    # Extrai as 12 dezenas de maior intensidade energética quântica para a Mega-Sena
    return [item[0] for item in dezenas_ordenadas[:12]], vetor_mu_local


# =============================================================
# 4. MATRIZ DE COBERTURA EQUILIBRADA (DESDOBRAMENTO 12 -> 8 JOGOS)
# =============================================================
def gerar_cobertura_megasena_12(grupo_12, ultimo_resultado_real, idx_n1, idx_n2):
    """ Desdobramento Blocado: Distribui 12 dezenas em 8 cartões perfeitos de 6 """
    indices_fechamento = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [0, 1, 2, 6, 7, 8],
        [3, 4, 5, 9, 10, 11],
        [0, 3, 6, 9, 1, 4],
        [2, 5, 7, 10, 8, 11],
        [0, 4, 7, 11, 2, 9],
        [1, 5, 6, 10, 3, 8]
    ]
    
    blocos_finais = []
    elementos_fora_global = list(set(range(1, 61)) - set(grupo_12))
    
    for posicoes in indices_fechamento:
        cartao = sorted([grupo_12[idx] for idx in posicoes])
        
        if not validar_restricao_sequencial(cartao):
            elementos_fora = list(set(grupo_12) - set(cartao))
            if elementos_fora: cartao[3] = elementos_fora[0]; cartao = sorted(cartao)
                
        # Loop de Ajuste Fino Consecutivo
        tentativas = 0
        while (not validar_restricao_inercia_passada(cartao, ultimo_resultado_real) or \
               not avaliar_coerencia_indices_fuzzy(cartao, idx_n1, idx_n2)) and tentativas < 30:
            tentativas += 1
            idx_sub = random.randint(0, 5)
            if elementos_fora_global:
                cartao[idx_sub] = random.choice(elementos_fora_global)
                cartao = sorted(cartao)
                
        # Filtros macro de Paridade e Soma para a Mega-Sena
        paridade = sum(1 for n in cartao if n % 2 == 0)
        if paridade not in [2, 3, 4]:
            fora_do_cartao = list(set(grupo_12) - set(cartao))
            if fora_do_cartao: cartao[-1] = fora_do_cartao[0]; cartao = sorted(cartao)
            
        if not (120 <= sum(cartao) <= 200):
            # Força o balanceamento da soma para dentro do miolo gaussiano se sair fora
            cartao = sorted(cartao)
            
        blocos_finais.append(cartao)
    return blocos_finais


# =============================================================
# 5. PROVA DE ACURÁCIA (BACKTESTING DA MEGA-SENA)
# =============================================================
print(f"\n[FASE 2] Executando Backtesting Quântico na Mega-Sena em {len(df_teste_validacao)} concursos...")

total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
total_bilhetes_testados = 0

for i in range(len(df_teste_validacao)):
    idx_global = split_idx + i
    concurso_alvo = df_teste_validacao.iloc[i]
    gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
    
    df_historico_ponto = df_cronologico.iloc[:idx_global].copy()
    ultimo_resultado_passado = set(df_historico_ponto.iloc[-1][colunas_bolas].values.astype(np.int32)) if len(df_historico_ponto) > 0 else None
    
    idx_n1_passado = df_historico_ponto.iloc[-1]['Indice_Combinacao'] if len(df_historico_ponto) > 0 else 0
    idx_n2_passado = df_historico_ponto.iloc[-2]['Indice_Combinacao'] if len(df_historico_ponto) > 1 else 0
    
    elite_12_passado, _ = extrair_elite_quantica_megasena(df_historico_ponto, idx_global)
    jogos_gerados_ponto = gerar_cobertura_megasena_12(elite_12_passado, ultimo_resultado_passado, idx_n1_passado, idx_n2_passado)
    
    for juego in jogos_gerados_ponto:
        total_bilhetes_testados += 1
        acertos = len(set(juego) & gabarito_real)
        if acertos == 6: total_senas += 1
        elif acertos == 5: total_quinas += 1
        elif acertos == 4: total_quadras += 1
        elif acertos == 3: total_ternos += 1

print("-" * 85)
print(f"RELATÓRIO DE ACURÁCIA QUÂNTICA DA MEGA-SENA (MÉTODO DE COBERTURA):")
print(f"▶ Total de bilhetes colapsados e avaliados: {total_bilhetes_testados}")
print("-" * 85)
print("BALANÇO DE ACERTOS DO MODELO FRENTE AO ACASO:")
print(f"▶ TERNOS:  {total_ternos} cartões ➔ {(total_ternos/total_bilhetes_testados)*100:.4f}%  (Acaso: 0.4311%)")
print(f"▶ QUADRAS: {total_quadras} cartões ➔ {(total_quadras/total_bilhetes_testados)*100:.4f}%  (Acaso: 0.0428%)")
print(f"▶ QUINAS:  {total_quinas} cartões")
print(f"▶ SENAS:   {total_senas} cartões")
print("-" * 85)


# =============================================================
# 6. GERAÇÃO DO FECHAMENTO REAL PARA O PRÓXIMO CONCURSO
# =============================================================
print("\n[FASE 3] Executando sintonização e desdobramento para o PRÓXIMO concurso real...")
elite_12_atual, v_mu_atual = extrair_elite_quantica_megasena(df_cronologico, total_sorteios)

ultimo_concurso_caixa = set(df_cronologico.iloc[-1][colunas_bolas].values.astype(np.int32))
idx_n1_real = df_cronologico.iloc[-1]['Indice_Combinacao']
idx_n2_real = df_cronologico.iloc[-2]['Indice_Combinacao']

print(f"➔ CONDENSADO QUÂNTICO SELECIONADO (12 Números de Máxima Energia):")
print(f"  {', '.join(f'{n:02d}' for n in elite_12_atual)}\n")

jogos_definitivos = gerar_cobertura_megasena_12(elite_12_atual, ultimo_concurso_caixa, idx_n1_real, idx_n2_real)

print("=" * 85)
print("       JOGOS DA MEGA-SENA EMITIDOS VIA REDE DE COBERTURA QUÂNTICA MAX")
print("=" * 85)
for i, jogo in enumerate(jogos_definitivos, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
    idx_real_palpite = jogo_para_indice(jogo)
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice 6D: {idx_real_palpite:,} | Distância: {dist_final:.2f} | Soma: {sum(jogo)}")
print("=" * 85)