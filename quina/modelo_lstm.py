import numpy as np
import pandas as pd
import math
import optuna
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.preprocessing import MinMaxScaler
from feature import df_resultado, cols_soma_dinamicas

# === 1. PREPARAÇÃO E CONFIGURAÇÃO ===
colunas_para_lstm = ['Indice', 'Frequencia', 'Distancia', 'quantidade_sequencia']
features = df_resultado[colunas_para_lstm + cols_soma_dinamicas].values

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
split = int(len(X1) * 0.9)
X1_train, X1_test = X1[:split], X1[split:]
X2_train, X2_test = X2[:split], X2[split:]
y_train, y_test = y[:split], y[split:]

# === 2. OTIMIZAÇÃO COM OPTUNA E LOG ===
def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    bottleneck_size = trial.suggest_int('bottleneck_size', 8, 32)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    input_seq = Input(shape=(16, X1.shape[2]))
    lstm_out = LSTM(lstm_units, activation='tanh')(input_seq)
    input_stat = Input(shape=(X2.shape[1],))
    bottleneck = Dense(bottleneck_size, activation='relu')(input_stat)
    
    merged = Concatenate()([lstm_out, bottleneck])
    out = Dense(32, activation='relu')(merged)
    output = Dense(1)(out)

    model = Model(inputs=[input_seq, input_stat], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    history = model.fit([X1_train, X2_train], y_train, epochs=20, batch_size=32, verbose=0)
    loss = model.evaluate([X1_test, X2_test], y_test, verbose=0)
    
    # Gravando log da rodada
    with open("log_optuna.txt", "a") as f:
        f.write(f"{datetime.now()} | Trial {trial.number} | Loss: {loss:.6f} | Params: {trial.params}\n")
        
    return loss

print("Iniciando busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best = study.best_params
print(f"Melhores parâmetros encontrados: {best}")

# === 3. MODELO FINAL ===
input_seq = Input(shape=(16, X1.shape[2]))
lstm_out = LSTM(best['lstm_units'], activation='tanh')(input_seq)
input_stat = Input(shape=(X2.shape[1],))
bottleneck = Dense(best['bottleneck_size'], activation='relu')(input_stat)

merged = Concatenate()([lstm_out, bottleneck])
out = Dense(32, activation='relu')(merged)
output = Dense(1, name="predicao_indice")(out)

model = Model(inputs=[input_seq, input_stat], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best['lr']), loss='mse')

model.fit([X1_train, X2_train], y_train, epochs=100, batch_size=32, validation_data=([X1_test, X2_test], y_test), verbose=1)

# === 4. VALIDADOR E PREDIÇÃO ===
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

def avaliar_acuracia(model, X1_test, X2_test, y_test_real):
    resultados = {"Terno": 0, "Quadra": 0, "Quina": 0, "Total": len(y_test_real)}
    pred_indices = model.predict([X1_test, X2_test], verbose=0).flatten()
    
    for i in range(len(pred_indices)):
        comb_prevista = set(indice_para_combinacao(80, 5, int(pred_indices[i])))
        comb_real = set(indice_para_combinacao(80, 5, int(y_test_real[i])))
        acertos = len(comb_prevista.intersection(comb_real))
        
        if acertos == 3: resultados["Terno"] += 1
        if acertos == 4: resultados["Quadra"] += 1
        if acertos == 5: resultados["Quina"] += 1
        
    return {k: (v / resultados["Total"]) * 100 for k, v in resultados.items()}

acuracia = avaliar_acuracia(model, X1_test, X2_test, y_test)
print("\n--- AVALIAÇÃO DE ACURÁCIA (TESTE) ---")
for cat, taxa in acuracia.items():
    print(f"{cat}: {taxa:.2f}%")

# === 5. PREDIÇÃO E LOG FINAL ===
pred = model.predict([X1[-1:], X2[-1:]])
proximo_indice = int(pred[0][0])
combinacao_prevista = sorted(list(indice_para_combinacao(80, 5, proximo_indice)))

print(f"\n--- SUGESTÃO DE JOGO ---")
print(f"Combinação Estimada: {combinacao_prevista}")

# Salvando resultado final
with open("historico_sugestoes.txt", "a") as f:
    f.write(f"{datetime.now()} | Sugestão: {combinacao_prevista} | Acuracia: {acuracia}\n")