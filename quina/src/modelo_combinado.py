import numpy as np
import pandas as pd
import math
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from feature import df_resultado, cols_soma_dinamicas

# === 1. PREPARAÇÃO DOS DADOS (LIMPEZA DE COLUNAS NÃO NUMÉRICAS) ===
# Garantir que removemos colunas que não podem ser convertidas para float
df_limpo = df_resultado.drop(columns=['Combinacao'], errors='ignore')

colunas_para_lstm = ['Indice', 'Frequencia', 'Distancia', 'quantidade_sequencia', 'Similaridade_Media']
features = df_limpo[colunas_para_lstm + cols_soma_dinamicas].values

scaler_t = MinMaxScaler()
scaler_e = MinMaxScaler()
X_t_scaled = scaler_t.fit_transform(features[:, :4]) 
X_e_scaled = scaler_e.fit_transform(features[:, 4:])

def create_hybrid_sequences(data_t, data_e, seq_length=16):
    X1, X2, y = [], [], []
    for i in range(len(data_t) - seq_length):
        X1.append(data_t[i : i + seq_length])
        X2.append(data_e[i + seq_length])
        y.append(features[i + seq_length, 0])
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_hybrid_sequences(X_t_scaled, X_e_scaled)

# Alinhamento: Remover as 16 linhas iniciais para coincidir com a janela da LSTM
df_alinhado = df_limpo.iloc[16:]
split = int(len(X1) * 0.9)

X1_train, X1_test = X1[:split], X1[split:]
X2_train, X2_test = X2[:split], X2[split:]
y_train, y_test = y[:split], y[split:]

# Preparar dados para CatBoost sem a coluna 'Combinacao'
X_cat = df_alinhado[colunas_para_lstm + cols_soma_dinamicas]
X_train_cat = X_cat.iloc[:split]
X_test_cat = X_cat.iloc[split:]
y_train_cat = y[:split]
y_test_cat = y[split:]

# === 2. TREINAMENTO MODELOS BASE ===
print("\nTreinando LSTM + Autoencoder...")
input_seq = Input(shape=(16, X1.shape[2]))
lstm_out = LSTM(64, activation='tanh')(input_seq)
input_stat = Input(shape=(X2.shape[1],))
bottleneck = Dense(16, activation='relu')(input_stat)
merged = Concatenate()([lstm_out, bottleneck])
model = Model(inputs=[input_seq, input_stat], outputs=Dense(1)(Dense(32, activation='relu')(merged)))
model.compile(optimizer='adam', loss='mse')
model.fit([X1_train, X2_train], y_train, epochs=50, batch_size=32, verbose=0)

print("Treinando CatBoost...")
best_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=False)
best_model.fit(X_train_cat, y_train_cat)

# === 3. STACKING (META-MODELO) ===
print("Treinando Meta-Modelo...")
pred_cat = best_model.predict(X_test_cat)
pred_ens = model.predict([X1_test, X2_test]).flatten()
pred_freq = np.full(len(pred_cat), df_resultado.sort_values('Frequencia', ascending=False)['Indice'].iloc[0])

X_meta = np.column_stack((pred_cat, pred_ens, pred_freq))
meta_model = CatBoostRegressor(iterations=100, depth=4, verbose=False)
meta_model.fit(X_meta, y_test_cat)

# === 4. PREVISÃO FINAL ===
last_cat = best_model.predict(X_test_cat.iloc[[-1]])
last_ens = model.predict([X1[-1:], X2[-1:]]).flatten()
last_freq = pred_freq[0]

X_final_meta = np.column_stack((last_cat, last_ens, [last_freq]))
indice_final = int(meta_model.predict(X_final_meta)[0])

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

# === 5. AVALIAÇÃO DE ACURÁCIA (META-MODELO / STACKING) ===
def avaliar_acuracia_stacking(meta_model, X_meta, y_test_real):
    resultados = {"Terno": 0, "Quadra": 0, "Quina": 0, "Total": len(y_test_real)}
    
    # Previsões do meta-modelo
    pred_meta = meta_model.predict(X_meta)
    
    # Garantir que y_test_real seja tratável como índice (array ou series)
    # Se for um DataFrame ou Series, usamos .iloc, se for array, usamos [i]
    is_pandas = hasattr(y_test_real, 'iloc')
    
    for i in range(len(pred_meta)):
        # Obtém o valor real de forma segura
        real_val = y_test_real.iloc[i] if is_pandas else y_test_real[i]
        
        comb_prevista = set(indice_para_combinacao(80, 5, int(pred_meta[i])))
        comb_real = set(indice_para_combinacao(80, 5, int(real_val)))
        
        acertos = len(comb_prevista.intersection(comb_real))
        
        if acertos == 3: resultados["Terno"] += 1
        if acertos == 4: resultados["Quadra"] += 1
        if acertos == 5: resultados["Quina"] += 1
        
    return {k: (v / resultados["Total"]) * 100 for k, v in resultados.items()}

print("\n--- AVALIAÇÃO DE PERFORMANCE (STACKING ENSEMBLE) ---")
metricas_stacking = avaliar_acuracia_stacking(meta_model, X_meta, y_test_cat)

for cat, taxa in metricas_stacking.items():
    if cat != "Total":
        print(f"Taxa de {cat}: {taxa:.2f}%")

# --- COMPARATIVO DE PERFORMANCE (IA vs SORTE) ---
print("\n--- VALIDAÇÃO: IA vs SORTE (PROBABILIDADE ALEATÓRIA) ---")

# Probabilidades teóricas de sorte aleatória
prob_aleatoria = {"Terno": 0.0467, "Quadra": 0.00156, "Quina": 0.00000416}

for cat, taxa_ia in metricas_stacking.items():
    if cat != "Total":
        prob_sorte = prob_aleatoria[cat]
        ratio = taxa_ia / prob_sorte if prob_sorte > 0 else 0
        print(f"{cat}: IA={taxa_ia:.4f}% | Sorte={prob_sorte:.4f}% | Desempenho={ratio:.2f}x o acaso")
        
combinacao_final = sorted(list(indice_para_combinacao(80, 5, indice_final)))
print(f"\nSugestão Final (Stacking): {combinacao_final}")
with open("historico_sugestoes_stacking.txt", "a") as f:
    f.write(f"\n--- {datetime.now()} ---\n")
    f.write(f"Sugestão: {combinacao_final}\n")
    
    # Escreve as métricas de acurácia no log
    f.write("Métricas de Performance:\n")
    for cat, taxa in metricas_stacking.items():
        if cat != "Total":
            f.write(f"- Taxa de {cat}: {taxa:.2f}%\n")
    
    f.write("-" * 30 + "\n")