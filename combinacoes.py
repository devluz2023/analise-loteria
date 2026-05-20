import itertools

# Isso NÃO carrega os 50 milhões na memória de uma vez
combinacoes = itertools.combinations(range(1, 61), 6)

# Para pegar uma combinação específica pelo índice,
# o ideal é usar lógica matemática em vez de converter para lista.
# Print the first 5 combinations
print("First 5 combinations:")
for i, combo in enumerate(combinacoes):
    if i >= 5:
        break
    print(combo)