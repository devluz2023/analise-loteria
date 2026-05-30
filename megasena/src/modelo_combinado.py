import pandas as pd
import numpy as np
import math
import optuna
from sklearn.linear_model import LinearRegression, BayesianRidge
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_squared_error

# === 1. PREPARAÇÃO ===
df_resultado = pd.read_excel("diagnostico_megasena_filtrado.xlsx")
df_limpo = df_resultado.drop(columns=['Combinacao', 'Indice'], errors='ignore')
target = df_resultado['Indice']

X_train, X_test, y_train, y_test = train_test_split(df_limpo, target, test_size=0.1, random_state=42)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_limpo)

def create_sequences(data, target, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df_limpo, target)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)

# === 2. FUNÇÕES DE APOIO ===
def indice_para_combinacao(n, k, index):
    comb = []; start = 1
    for i in range(k):
        for val in range(start, n + 1):
            count = math.comb(n - val, k - 1 - i)
            if index < count: comb.append(val); start = val + 1; break
            else: index -= count
    return comb

# AQUI FOI CORRIGIDO: Adicionado 'is_lstm' como argumento opcional
def avaliar(preds, y_test, nome, is_lstm=False):
    preds = preds.flatten()
    total = len(y_test)
    acertos = {"Terno": 0, "Quadra": 0, "Quina": 0, "Sena": 0}
    
    for pred, actual in zip(preds, y_test):
        p = set(indice_para_combinacao(60, 6, int(max(0, min(float(pred), 50063859)))))
        r = set(indice_para_combinacao(60, 6, int(actual)))
        inter = len(p.intersection(r))
        if inter >= 3:
            nome_acerto = ["", "", "", "Terno", "Quadra", "Quina", "Sena"][inter]
            acertos[nome_acerto] += 1
            
    print(f"\n--- Performance: {nome} ({total} testes) ---")
    for k, v in acertos.items():
        print(f"{k}: {v} acertos ({(v/total)*100:.4f}%)")

# === 3. TREINAMENTO DOS 4 MODELOS ===
print("Treinando modelos (Linear, Bayesiano, CatBoost, LSTM)...")

# 1. Linear
model_lr = LinearRegression().fit(X_train, y_train)

# 2. Bayesiano
model_bayes = BayesianRidge().fit(X_train, y_train)

# 3. CatBoost (Otimizado com Optuna)
def objective_cb(trial):
    params = {'iterations': trial.suggest_int('iterations', 100, 500), 'depth': trial.suggest_int('depth', 4, 8)}
    model = CatBoostRegressor(**params, verbose=False).fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test))

study = optuna.create_study(direction='minimize')
study.optimize(objective_cb, n_trials=10)
model_cb = CatBoostRegressor(**study.best_params, verbose=False).fit(X_train, y_train)

# 4. LSTM (Otimizado)
model_lstm = Sequential([Input(shape=(X_seq.shape[1], X_seq.shape[2])), LSTM(64), Dense(1)])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_seq_train, y_seq_train, epochs=30, verbose=0)

# === 4. COMPARAÇÃO FINAL ===
avaliar(model_lr.predict(X_test), y_test, "Regressão Linear")
avaliar(model_bayes.predict(X_test), y_test, "Regressão Bayesiana")
avaliar(model_cb.predict(X_test), y_test, "CatBoost")
avaliar(model_lstm.predict(X_seq_test, verbose=0), y_seq_test, "LSTM", is_lstm=True)