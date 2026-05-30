import numpy as np
import pandas as pd
import math
import pulp
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

# === 1. LEITURA E PREPARAÇÃO ===
df_resultado = pd.read_excel("diagnostico_quina_filtrado.xlsx")
df_limpo = df_resultado.drop(columns=['Combinacao'], errors='ignore')

cols_base = ['Indice', 'Frequencia', 'Distancia', 'quantidade_sequencia']
if 'Similaridade_Media' in df_limpo.columns:
    cols_base.append('Similaridade_Media')
cols_soma = [c for c in df_limpo.columns if c.endswith('_Soma')]
features = df_limpo[cols_base + cols_soma].values

X_t_scaled = MinMaxScaler().fit_transform(features[:, :4]) 
X_e_scaled = MinMaxScaler().fit_transform(features[:, 4:])

def create_hybrid_sequences(data_t, data_e, seq_length=16):
    X1, X2, y = [], [], []
    for i in range(len(data_t) - seq_length):
        X1.append(data_t[i : i + seq_length])
        X2.append(data_e[i + seq_length])
        y.append(features[i + seq_length, 0])
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_hybrid_sequences(X_t_scaled, X_e_scaled)
split = int(len(X1) * 0.9)

# === 2. TREINAMENTO MODELOS ===
input_seq = Input(shape=(16, X1.shape[2]))
lstm_out = LSTM(64, activation='tanh')(input_seq)
input_stat = Input(shape=(X2.shape[1],))
merged = Concatenate()([lstm_out, Dense(16, activation='relu')(input_stat)])
model = Model(inputs=[input_seq, input_stat], outputs=Dense(1)(Dense(32, activation='relu')(merged)))
model.compile(optimizer='adam', loss='mse')
model.fit([X1[:split], X2[:split]], y[:split], epochs=20, verbose=0)

best_model = CatBoostRegressor(iterations=300, verbose=False)
best_model.fit(df_limpo.iloc[16:16+split].drop(columns=['Indice'], errors='ignore'), y[:split])

meta_model = CatBoostRegressor(iterations=100, verbose=False)
meta_model.fit(np.column_stack((best_model.predict(df_limpo.iloc[16:16+split].drop(columns=['Indice'], errors='ignore')), 
                                model.predict([X1[:split], X2[:split]]).flatten(), 
                                np.full(split, df_resultado.sort_values('Frequencia', ascending=False)['Indice'].iloc[0]))), y[:split])

# === 3. FUNÇÕES DE APOIO E OTIMIZAÇÃO ===
def indice_para_combinacao(n, k, index):
    comb = []
    start = 1
    for i in range(k):
        for val in range(start, n + 1):
            count = math.comb(n - val, k - 1 - i)
            if index < count:
                comb.append(val); start = val + 1; break
            else: index -= count
    return comb

def otimizar_jogo_pulp(indice_meta_modelo):
    media_alvo = (indice_meta_modelo / 100000000) * 80
    prob = pulp.LpProblem("OtimizacaoQuina", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("d", range(1, 81), cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in range(1, 81)]) == 5
    prob += pulp.lpSum([i * x[i] for i in range(1, 81)]) >= 170
    prob += pulp.lpSum([i * x[i] for i in range(1, 81)]) <= 230
    prob += pulp.lpSum([x[i] for i in range(1, 81) if i % 2 == 0]) >= 2
    prob += pulp.lpSum([x[i] for i in range(1, 81) if i % 2 == 0]) <= 3
    prob += pulp.lpSum([x[i] * abs(i - media_alvo) for i in range(1, 81)])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sorted([i for i in range(1, 81) if pulp.value(x[i]) == 1])

# === 4. AVALIAÇÃO DE PERFORMANCE ===
def avaliar_sistema(y_test, X1_test, X2_test, df_test_rows):
    resultados = {"Terno": 0, "Quadra": 0, "Quina": 0}
    for i in range(len(y_test)):
        # IA prevê, PuLP otimiza
        idx = int(meta_model.predict(np.column_stack((best_model.predict(df_test_rows.iloc[[i]].drop(columns=['Indice'], errors='ignore')), 
                                                     model.predict([X1_test[[i]], X2_test[[i]]]).flatten(), [features[0,0]])))[0])
        jogo_ia_pulp = set(otimizar_jogo_pulp(idx))
        jogo_real = set(indice_para_combinacao(80, 5, int(y_test[i])))
        acertos = len(jogo_ia_pulp.intersection(jogo_real))
        if acertos >= 3: resultados["Terno" if acertos==3 else "Quadra" if acertos==4 else "Quina"] += 1
    return resultados

# Execução
print("\n--- PERFORMANCE DO SISTEMA IA + PuLP ---")
res = avaliar_sistema(y[split:], X1[split:], X2[split:], df_limpo.iloc[16+split:])
for k, v in res.items(): print(f"{k}: {v} acertos")

# Previsão final
idx_final = int(meta_model.predict(np.column_stack((best_model.predict(df_limpo.iloc[[-1]].drop(columns=['Indice'], errors='ignore')), 
                                                   model.predict([X1[-1:], X2[-1:]]).flatten(), [features[0,0]])))[0])
print(f"\nSugestão Final: {otimizar_jogo_pulp(idx_final)}")