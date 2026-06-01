import math

# Ajuste: A Lotofácil possui colunas de Bola1 a Bola15
colunas_bolas = [f'Bola{i}' for i in range(1, 16)]

# Ajuste: O universo é de 25 dezenas e escolhem-se 15
TOTAL_DEZENAS = 25
QUANTIDADE_ESCOLHAS = 15

# O cálculo do total de combinações possíveis
TOTAL_POSSIVEL = math.comb(TOTAL_DEZENAS, QUANTIDADE_ESCOLHAS)

print(f"Total de combinações possíveis na Lotofácil: {TOTAL_POSSIVEL:,}")