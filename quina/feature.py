import numpy as np
import pandas as pd
import requests
import io
import math
import os
from collections import Counter
from matriz_frequencia_quina import gerar_matriz_quina
from get_data import df, colunas_bolas, TOTAL_POSSIVEL
from itertools import combinations
df_matriz = gerar_matriz_quina(df)
df_matriz.to_excel("matriz_ausencia_quina.xlsx", index=False)

# Prepara mapas estatísticos convertendo tudo para float (trata strings e notação científica)
mapas_estatisticos = {}
for col in df_matriz.columns:
    if col != 'Dezena':
        # Converte a coluna inteira para numérico, forçando erro se não for possível
        valores = pd.to_numeric(df_matriz[col], errors='coerce').fillna(0)
        mapas_estatisticos[col] = dict(zip(df_matriz['Dezena'], valores))



# === 2. FUNÇÕES ESSENCIAIS ===
def calcular_indice(comb_list):
    comb_list = sorted(comb_list)
    index = 0
    start = 1
    for i in range(5):
        for val in range(start, comb_list[i]):
            index += math.comb(80 - val, 5 - 1 - i)
        start = comb_list[i] + 1
    return index

def calcular_distancia_circular(idx_atual, idx_proximo):
    dist = abs(idx_proximo - idx_atual)
    return min(dist, TOTAL_POSSIVEL - dist)

def maior_sequencia_circular(comb_list):
    comb = sorted([int(n) for n in comb_list])
    grupos = []
    i = 0
    while i < len(comb):
        g = [comb[i]]
        while i + 1 < len(comb) and comb[i+1] == comb[i] + 1:
            g.append(comb[i+1])
            i += 1
        grupos.append(g)
        i += 1
    if len(grupos) > 1 and 1 in grupos[0] and 80 in grupos[-1]:
        grupos[0] = grupos[-1] + grupos[0]; grupos.pop()
    return max([len(g) for g in grupos])

# === 3. PROCESSAMENTO E INTEGRAÇÃO ===
df['Indice'] = df.apply(lambda row: calcular_indice([int(row[c]) for c in colunas_bolas]), axis=1)
ultimo_indice = df['Indice'].iloc[-1]

# Mapeamento de Elite
historico = [tuple(sorted([int(row[col]) for col in colunas_bolas])) for _, row in df.iterrows()]
freq_map = Counter(historico)
df_elite = pd.DataFrame([{'Indice': calcular_indice(c), 'Combinacao': c, 'Frequencia': f} for c, f in freq_map.items()])

# Métricas Originais
df_elite['Distancia'] = df_elite['Indice'].apply(lambda x: calcular_distancia_circular(x, ultimo_indice))
df_elite['quantidade_sequencia'] = df_elite['Combinacao'].apply(maior_sequencia_circular)

# Integração DINÂMICA de TODAS as colunas da matriz
for nome_coluna, mapa in mapas_estatisticos.items():
    df_elite[f'{nome_coluna}_Soma'] = df_elite['Combinacao'].apply(lambda c: sum(mapa.get(n, 0) for n in c))
    df_elite[f'{nome_coluna}_Max'] = df_elite['Combinacao'].apply(lambda c: max([mapa.get(n, 0) for n in c], default=0))

# === 4. RESULTADO ===
df_resultado = df_elite[df_elite['Frequencia'] > 0].sort_values(by='Distancia')

print("\n=== DIAGNÓSTICO INTEGRADO COMPLETO ===")
print(df_resultado.head(15).to_string())


# === EXIBIÇÃO FORMATADA COM TODAS AS SOMAS ===

# Identifica automaticamente todas as colunas de soma que criamos
# === DEFINIÇÃO DAS COLUNAS FIXAS ===
# === DEFINIÇÃO DAS COLUNAS ===
colunas_fixas = [
    'Indice', 
    'Combinacao', 
    'Frequencia', 
    'Distancia', 
    'quantidade_sequencia'
]

# Filtra apenas colunas que terminam com '_Soma' E que são numéricas (excluindo as colunas indesejadas)
# Lista de colunas para EXCLUIR explicitamente
colunas_para_remover = [
    'Total_Streaks_Observed_Soma', 
    'Total_Absence_Draws_Soma', 
    'Average_Streak_Length_Soma',
    'Desvio_Soma',
    'Prob_Nao_Sair_Empirica_1_Concurso_Soma',
    'Prob_Nao_Sair_Empirica_247_Concursos_Soma',
    'Máximo Consecutivo_Soma',
    'Total de Sequências (qualquer tamanho)_Soma',
    'Total de Sequências (> 1)_Soma',
    'Máximo de Sorteios Sem Sair_Soma',
    'Ausencia_Actual_Concursos_Soma'
]

# Seleciona todas as colunas '_Soma', mas remove as que você não quer
cols_soma_dinamicas = [
    c for c in df_elite.columns 
    if c.endswith('_Soma') and c not in colunas_para_remover
]

# Junta tudo
colunas_finais = colunas_fixas + cols_soma_dinamicas

# === EXIBIÇÃO ===
print("\n=== DIAGNÓSTICO FILTRADO (SOMENTE ÍNDICES DE JEJUM) ===")
print(df_resultado[colunas_finais].head(15).to_string(index=False))

# Salva apenas o que você pediu
df_resultado[colunas_finais].to_excel("diagnostico_quina_filtrado.xlsx", index=False)


