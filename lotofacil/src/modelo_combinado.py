import pandas as pd
import numpy as np
import pulp
import random
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import GridSearchCV
# === 1. CARREGAMENTO E PREPARAÇÃO ===
df = pd.read_excel("diagnostico_lotofacil_filtrado.xlsx")
df_limpo = df.select_dtypes(include=[np.number]).fillna(0)

# Preparar dados para LSTM e outros modelos
colunas_dezenas = df_limpo.columns[1:16]
historico_real = [set(row[colunas_dezenas].astype(int).tolist()) for _, row in df_limpo.iterrows()]
X_data = df_limpo.values
y_data = df_limpo[colunas_dezenas].values

# Treino CatBoost
X = df_limpo.drop(columns=[df_limpo.columns[0]])
y = df_limpo[colunas_dezenas].sum(axis=1)
modelo_cb = CatBoostRegressor(iterations=100, verbose=0).fit(X, y)

# Treino LSTM
def criar_sequencias(X, y, janela=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - janela):
        X_seq.append(X[i:(i + janela)])
        y_seq.append(y[i + janela])
    return np.array(X_seq), np.array(y_seq)

X_lstm, y_lstm = criar_sequencias(X_data, y_data, janela=5)

def treinar_modelo_lstm(X_train, y_train):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(15) 
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)
    return model

modelo_lstm_obj = treinar_modelo_lstm(X_lstm, y_lstm)

# === 2. MODELOS ===
def modelo_po():
    prob = pulp.LpProblem("PO", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("d", range(1, 26), cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in range(1, 26)]) == 15
    prob += pulp.lpSum([i * x[i] for i in range(1, 26)]) >= 175
    prob += pulp.lpSum([i * x[i] for i in range(1, 26)]) <= 215
    prob += pulp.lpSum([x[i] * random.random() for i in range(1, 26)])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return set([int(i) for i in range(1, 26) if pulp.value(x[i]) == 1])

modelo_base = CatBoostRegressor(verbose=0)
param_grid = {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8]}
grid_search = GridSearchCV(estimator=modelo_base, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X, y)
modelo_cb_final = grid_search.best_estimator_ # Este é o modelo já treinado!

# 2. FUNÇÃO SIMPLES QUE APENAS FAZ A PREVISÃO
def modelo_catboost():
    # Usa o modelo já treinado globalmente
    pesos = modelo_cb_final.predict(X.iloc[[-1]]) 
    sugestao = np.argsort(pesos[0])[-15:] + 1
    return set(sugestao)


def modelo_lstm():
    dados_globais = df_limpo.values
    ultima_sequencia = dados_globais[-5:].reshape(1, 5, dados_globais.shape[1])
    previsao = modelo_lstm_obj.predict(ultima_sequencia, verbose=0)
    sugestao = np.argsort(previsao[0])[-15:] + 1
    return set(sugestao)

# === 3. ARENA DE COMPARAÇÃO (Corrigida) ===
class Arena:
    def __init__(self):
        # ADICIONADO 'LSTM' NA ESTRUTURA
        self.stats = {
            'PO': {12:0, 13:0, 14:0, 15:0}, 
            'CatBoost': {12:0, 13:0, 14:0, 15:0},
            'LSTM': {12:0, 13:0, 14:0, 15:0}
        }
        self.total = 0

    def registrar(self, modelo, acertos):
        if modelo in self.stats and acertos in self.stats[modelo]:
            self.stats[modelo][acertos] += 1

arena = Arena()

# === 4. EXECUÇÃO DO BACKTESTING ===
print(f"Analisando {len(historico_real)} sorteios...")

for sorteio_real in historico_real:
    arena.total += 1
    jogos = {'PO': modelo_po(), 'CatBoost': modelo_catboost(), 'LSTM': modelo_lstm()}
    
    for nome, jogo in jogos.items():
        acertos = len(jogo.intersection(sorteio_real))
        if acertos >= 12:
            arena.registrar(nome, acertos)

# === 5. RESULTADO FINAL ===
print("\n" + "="*65)
print(f"{'RELATÓRIO DE PERFORMANCE (Amostra: ' + str(arena.total) + ')':^65}")
print("="*65)
print(f"{'Modelo':<12} | {'12 Ac':<6} | {'13 Ac':<6} | {'14 Ac':<6} | {'15 Ac':<6} | {'Eficiência %':<10}")
print("-"*65)

for modelo, premios in arena.stats.items():
    c12, c13, c14, c15 = premios[12], premios[13], premios[14], premios[15]
    total_premios = c12 + c13 + c14 + c15
    eficiencia = (total_premios / arena.total) * 100 if arena.total > 0 else 0
    print(f"{modelo:<12} | {c12:<6} | {c13:<6} | {c14:<6} | {c15:<6} | {eficiencia:>9.2f}%")


# === 6. LÓGICA PARA ESCOLHER E GERAR O MELHOR JOGO ===
def exibir_melhor_resultado(arena):
    # Sistema de pontuação: 15 acertos é o topo da pirâmide
    scores = {}
    for modelo, premios in arena.stats.items():
        # Cálculo do score: (15*1000) + (14*100) + (13*10) + (12*1)
        score = (premios[15]*1000) + (premios[14]*100) + (premios[13]*10) + (premios[12]*1)
        scores[modelo] = score
    
    # Identificar o vencedor
    melhor_modelo = max(scores, key=scores.get)
    
    print("\n" + "="*65)
    print(f"MODELO VENCEDOR: {melhor_modelo} (Score: {scores[melhor_modelo]})")
    print("="*65)
    
    # Gerar dezenas do modelo vencedor
    if melhor_modelo == 'PO':
        dezenas = modelo_po()
    elif melhor_modelo == 'CatBoost':
        dezenas = modelo_catboost()
    else: # LSTM
        dezenas = modelo_lstm()
        
    print(f"Dezenas sugeridas pelo {melhor_modelo} para o próximo sorteio:")
    print(sorted(list(dezenas)))
    print("="*65)

# Executa a função final
exibir_melhor_resultado(arena)

# Executa a função final
exibir_melhor_resultado(arena)