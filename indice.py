import pandas as pd
from math import comb
from get_data import df_mega_sena

# 1. Identificar as colunas das dezenas (certifique-se que os nomes estão corretos no seu Excel)
# Com base no seu comando, as colunas geralmente são 'Bola1', 'Bola2', etc.
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']

def encontrar_posicao_itertools(linha):
    """
    Calcula a posição exata que uma combinação ocuparia no
    gerador itertools.combinations(range(1, 61), 6)
    """
    # Extrair os números da linha e garantir que estão ordenados
    dezenas = sorted([int(n) for n in linha[colunas_bolas]])

    n = 60
    k = 6
    posicao = 0
    valor_anterior = 1

    for i, valor in enumerate(dezenas):
        # Somamos as combinações que foram "puladas"
        for v in range(valor_anterior, valor):
            posicao += comb(n - v, k - 1 - i)
        valor_anterior = valor + 1

    return posicao + 1

# 2. Criar a nova coluna no DataFrame
print("Calculando o grupo/índice de cada sorteio...")
df_mega_sena['Indice_Combinacao'] = df_mega_sena.apply(encontrar_posicao_itertools, axis=1)

# 3. Exibir o resultado final com a nova coluna
print(df_mega_sena[['Concurso', 'Indice_Combinacao'] + colunas_bolas].head())