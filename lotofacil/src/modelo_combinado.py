import pandas as pd
import numpy as np
import pulp
import random

# === 1. CARREGAMENTO E NORMALIZAÇÃO ===
df = pd.read_excel("diagnostico_lotofacil_filtrado.xlsx")
cols_bolas = df.select_dtypes(include=[np.number]).columns[1:16] 

# FORÇAMOS que todos os dados do histórico sejam inteiros (int)
historico_real = []
for _, row in df.iterrows():
    bolas = set(row[cols_bolas].dropna().astype(int).tolist())
    historico_real.append(bolas)

# === 2. MODELO PO ===
def modelo_po_lotofacil():
    prob = pulp.LpProblem("PO_Lotofacil", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("d", range(1, 26), cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in range(1, 26)]) == 15
    prob += pulp.lpSum([i * x[i] for i in range(1, 26)]) >= 175
    prob += pulp.lpSum([i * x[i] for i in range(1, 26)]) <= 215
    # Objetivo aleatório
    prob += pulp.lpSum([x[i] * random.random() for i in range(1, 26)])
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=1)
    prob.solve(solver)
    return set([int(i) for i in range(1, 26) if pulp.value(x[i]) == 1])

# === 3. AGENTE COM CONTADOR CORRIGIDO ===
class AgenteRL:
    def __init__(self):
        # Dicionários de contagem inicializados
        self.contadores = {12: 0, 13: 0, 14: 0, 15: 0}
        self.total_testes = 0

    def registrar_acerto(self, n_acertos):
        self.total_testes += 1
        if n_acertos in self.contadores:
            self.contadores[n_acertos] += 1
            print(f"BINGO! O modelo acertou {n_acertos} números!")

# === 4. EXECUÇÃO ===
agente = AgenteRL()
print(f"Iniciando análise em {len(historico_real)} concursos...")

for sorteio_real in historico_real:
    sugestao = modelo_po_lotofacil()
    
    # A MÁGICA: Aqui garantimos que ambos são sets de inteiros
    acertos = len(sugestao.intersection(sorteio_real))
    
    if acertos >= 12:
        agente.registrar_acerto(acertos)

# === 5. RESULTADO ===
print("\n" + "="*40)
print("TOTAL DE PRÉMIOS IDENTIFICADOS:")
print("="*40)
for premio, qtd in agente.contadores.items():
    print(f"Prémios de {premio} acertos: {qtd}")
print(f"Total de sorteios analisados: {agente.total_testes}")