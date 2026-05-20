import random
from itertools import combinations
import pandas as pd
from get_data import df_mega_sena
from frequencia import repeticoes, sorteios

# === IMPORTAÇÃO DO SEU SCRIPT INDICE.PY (SEM ALTERAÇÕES NELE) ===
from indice import encontrar_posicao_itertools 

# =============================================================
# 1. CÁLCULO DAS PROBABILIDADES REAIS (BASE CONCURSOS)
# =============================================================

total_concursos = len(df_mega_sena)
print(f"\nCalculando matriz de probabilidades com base em {total_concursos} concursos...")

probabilidades = {
    6: 0.0,  
    5: len(repeticoes[5]) / total_concursos,
    4: len(repeticoes[4]) / total_concursos,
    3: len(repeticoes[3]) / total_concursos,
    2: len(repeticoes[2]) / total_concursos
}

print("-" * 60)
print("PROBABILIDADES EMPÍRICAS ENCONTRADAS NO SEU HISTÓRICO:")
print(f"▶ Chance de Sena repetida: {probabilidades[6]:.4f}%")
print(f"▶ Chance de Quina repetida: {probabilidades[5] * 100:.4f}%")
print(f"▶ Chance de Quadra repetida: {probabilidades[4] * 100:.4f}%")
print(f"▶ Chance de Terno repetido: {probabilidades[3] * 100:.4f}%")
print("-" * 60)

senas_historicas = set(tuple(sorted(list(jogo))) for jogo in sorteios.values())
quinas_historicas = set(repeticoes[5].keys())
quadras_historicas = set(repeticoes[4].keys())
ternos_historicas = set(repeticoes[3].keys())


# =============================================================
# 2. MOTOR DE VALIDAÇÃO
# =============================================================

def jogo_e_valido_estritamente(novo_jogo):
    novo_jogo_tuple = tuple(novo_jogo)
    
    if novo_jogo_tuple in senas_historicas:
        return False
        
    for quina in combinations(novo_jogo, 5):
        if quina in quinas_historicas:
            if random.random() > probabilidades[5]:
                return False
                
    contagem_quadras = 0
    for quadra in combinations(novo_jogo, 4):
        if quadra in quadras_historicas:
            contagem_quadras += 1
    
    if contagem_quadras > 0:
        if random.random() > probabilidades[4]:
            return False

    contagem_ternos = 0
    for terno in combinations(novo_jogo, 3):
        if terno in ternos_historicas:
            contagem_ternos += 1
            
    if contagem_ternos < 1 or contagem_ternos > 3:
        return False
        
    return True


# =============================================================
# 3. GERAÇÃO PROPORCIONAL
# =============================================================

def gerar_jogos_por_frequencia(quantidade=5):
    jogos_finais = []
    tentativas = 0
    
    while len(jogos_finais) < quantidade:
        tentativas += 1
        palpite = sorted(random.sample(range(1, 61), 6))
        
        if jogo_e_valido_estritamente(palpite):
            jogos_finais.append(palpite)
            
    return jogos_finais, tentativas


# =============================================================
# 4. EXIBIÇÃO DOS PALPITES COM OS ÍNDICES CALCULADOS DO INDICE.PY
# =============================================================

N_JOGOS = 5
jogos_gerados, total_testes = gerar_jogos_por_frequencia(N_JOGOS)

print("\n" + "=" * 70)
print("     GERADOR PROBABILÍSTICO + CÁLCULO DE ÍNDICE (INDICE.PY)")
print("=" * 70)
print(f"Palpites processados até alinhar com o histórico: {total_testes}")
print("-" * 70)

for i, jogo in enumerate(jogos_gerados, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    
    # === SOLUÇÃO: Criamos uma linha nativa do Pandas (Series) ===
    # Isso simula perfeitamente uma linha do Excel para o 'indice.py' aceitar o filtro
    linha_pandas = pd.Series(
        data=jogo, 
        index=['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    )
    
    # Executa a função do seu arquivo sem dar erro de tipo
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    
    print(f"Jogo {i}: [ {jogo_formatado} ] -> Índice: {indice_calculado:,}")

print("=" * 70)
print("Índices gerados automaticamente usando a lógica do seu 'indice.py'.")
print("=" * 70)