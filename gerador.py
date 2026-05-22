import random
from itertools import combinations
import pandas as pd
import numpy as np
from get_data import df_mega_sena
from frequencia import repeticoes, sorteios
from matriz_frequencia import gerar_matriz

# =============================================================
# 1. MAPEAMENTO MACRO DE PROBABILIDADES DA MATRIZ DE AUSÊNCIA
# =============================================================
geteracao_matriz = gerar_matriz()  # Atualiza o teu Excel automaticamente
df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")

# Filtrar quais são as colunas numéricas de jejum (1 até 108 ou mais)
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

# Dicionários para guardar a inteligência analítica do teu Excel
atraso_atual_por_dezena = {}
distribuicao_global_jejuns = {int(col): 0 for col in colunas_jejum}

# Passo A: Capturar o atraso de hoje de cada dezena e somar os totais de cada coluna
for index, row in df_ausencia.iterrows():
    dezena = int(row['Dezena'])
    atraso_atual_por_dezena[dezena] = int(row['Ausencia_Actual_Concursos'])
    
    for col in colunas_jejum:
        tamanho_j = int(col)
        distribuicao_global_jejuns[tamanho_j] += int(row[col])

# Passo B: Calcular o total absoluto de sequências gravadas na história
total_absoluto_sequencias = sum(distribuicao_global_jejuns.values())

# Passo C: Criar a tabela de probabilidade real para cada tamanho de jejum (1 a 108)
probabilidade_tamanho_jejum = {}
for tamanho_j, total_ocorrencias in distribuicao_global_jejuns.items():
    # Evita divisão por zero e define a chance empírica daquele tamanho de jejum específico
    probabilidade_tamanho_jejum[tamanho_j] = total_ocorrencias / total_absoluto_sequencias

print("\n" + "=" * 75)
print("     CURVA DE PROBABILIDADE HISTÓRICA DO COMPRIMENTO DE JEJUM")
print("=" * 75)
print(f"Total de ciclos de jejum analisados no Excel: {total_absoluto_sequencias:,}")
print(f"▶ Probabilidade de fechar Jejum tamanho 1: {probabilidade_tamanho_jejum.get(1, 0)*100:.2f}% (Alta)")
print(f"▶ Probabilidade de fechar Jejum tamanho 2: {probabilidade_tamanho_jejum.get(2, 0)*100:.2f}% (Alta)")
print(f"▶ Probabilidade de fechar Jejum tamanho 10: {probabilidade_tamanho_jejum.get(10, 0)*100:.2f}% (Média)")
# Busca dinamicamente o maior valor da lista para printar a probabilidade da cauda longa
maior_j_gravado = max(probabilidade_tamanho_jejum.keys())
print(f"▶ Probabilidade de fechar Jejum tamanho {maior_j_gravado}: {probabilidade_tamanho_jejum.get(maior_j_gravado, 0)*100:.4f}% (Raríssima)")
print("=" * 75)


# =============================================================
# 2. PROBABILIDADES ESTOCÁSTICAS DE REPETIÇÃO DE GRUPOS
# =============================================================
total_concursos = len(df_mega_sena)
probabilidades_repeticao = {
    6: 0.0,  
    5: len(repeticoes[5]) / total_concursos,
    4: len(repeticoes[4]) / total_concursos,
    3: len(repeticoes[3]) / total_concursos
}

senas_historicas = set(tuple(sorted(list(jogo))) for jogo in sorteios.values())
quinas_historicas = set(repeticoes[5].keys())
quadras_historicas = set(repeticoes[4].keys())
ternos_historicas = set(repeticoes[3].keys())


# =============================================================
# 3. MOTOR DE VALIDAÇÃO FIEL ÀS PROBABILIDADES DAS COLUNAS (1 A 108)
# =============================================================
def jogo_e_valido_estritamente(novo_jogo):
    novo_jogo_tuple = tuple(novo_jogo)
    
    # ---------------------------------------------------------------------
    # BARREIRA 1: ADERÊNCIA À CURVA MATRICIAL DE PROBABILIDADE DE JEJUM
    # ---------------------------------------------------------------------
    probabilidades_das_6_dezenas = []
    
    for num in novo_jogo:
        # Se a dezena sair hoje, o ciclo fecha no tamanho atual dela + 1
        jejum_futuro = atraso_atual_por_dezena[num] + 1
        
        # Puxa do nosso dicionário qual é a probabilidade geral de um jejum fechar com esse tamanho
        chance_desse_jejum = probabilidade_tamanho_jejum.get(jejum_futuro, 0)
        
        probabilidades_das_6_dezenas.append(chance_desse_jejum)
        
    # Critério de Rejeição 1: Se o jogo contiver qualquer dezena cujo tamanho de
    # jejum futuro seja uma anomalia absoluta (chance menor que 0.05% ou 0.0005)
    # permitindo apenas se for jejum ultra baixo (1 ou 2) que são zonas de repetição rápida
    for p, num in zip(probabilidades_das_6_dezenas, novo_jogo):
        if p < 0.0005 and (atraso_atual_por_dezena[num] + 1) > 2:
            return False # Descarta na hora por ser um tamanho de jejum improvável de fechar hoje

    # Critério de Rejeição 2: O equilíbrio coletivo das probabilidades.
    # Em média, um jogo normal sorteado é composto por dezenas que flutuam em posições
    # confortáveis da curva. Se o produto ou a média for perigosamente baixa, o jogo é descartado.
    if np.mean(probabilidades_das_6_dezenas) < 0.015:
        return False

    # ---------------------------------------------------------------------
    # BARREIRA 2: FILTRO DE REPETIÇÃO ESTOCÁSTICA (QUADRAS/TERNOS)
    # ---------------------------------------------------------------------
    if novo_jogo_tuple in senas_historicas:
        return False
        
    for quina in combinations(novo_jogo, 5):
        if quina in quinas_historicas and random.random() > probabilidades_repeticao[5]:
            return False
                
    contagem_quadras = sum(1 for quadra in combinations(novo_jogo, 4) if quadra in quadras_historicas)
    if contagem_quadras > 0 and random.random() > probabilidades_repeticao[4]:
        return False

    contagem_ternos = sum(1 for terno in combinations(novo_jogo, 3) if terno in ternos_historicas)
    if contagem_ternos < 1 or contagem_ternos > 3:
        return False
        
    return True