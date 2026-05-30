import pandas as pd
import numpy as np
import math
import optuna
import pulp
from sklearn.linear_model import LinearRegression, BayesianRidge
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# === 1. PREPARAÇÃO E SINCRONIZAÇÃO ===
df_resultado = pd.read_excel("diagnostico_megasena_filtrado.xlsx")
df_limpo = df_resultado.drop(columns=['Combinacao', 'Indice'], errors='ignore')
target = df_resultado['Indice']

def create_sequences(data, target, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i : i + seq_length].values)
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df_limpo, target)
indices = np.arange(len(X_seq))
train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=False)

X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
y_seq_train, y_seq_test = y_seq[train_idx], y_seq[test_idx]
X_train, X_test = df_limpo.iloc[train_idx + 5], df_limpo.iloc[test_idx + 5]
y_train, y_test = target.iloc[train_idx + 5], target.iloc[test_idx + 5]

# === 2. FUNÇÕES DE APOIO ===
def avaliar(preds, y_target, nome):
    preds = preds.flatten()
    print(f"\n--- Performance: {nome} (MSE: {mean_squared_error(y_target, preds):.2f}) ---")

def otimizar_com_pulp(indice_alvo):
    media_alvo = (indice_alvo / 50063859) * 60
    prob = pulp.LpProblem("Otimizacao_Final", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("d", range(1, 61), cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in range(1, 61)]) == 6
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) >= 160
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) <= 240
    prob += pulp.lpSum([x[i] * abs(i - media_alvo) for i in range(1, 61)])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sorted([i for i in range(1, 61) if pulp.value(x[i]) == 1])

# === 3. OTIMIZAÇÃO HÍBRIDA ===
# Grid Search para Bayes
grid_bayes = GridSearchCV(BayesianRidge(), {'alpha_1': [1e-6, 1e-4], 'lambda_1': [1e-6, 1e-4]}, cv=3)
grid_bayes.fit(X_train, y_train)

# Optuna para Ensemble
def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 32, 64)
    cb = CatBoostRegressor(iterations=200, verbose=False).fit(X_train, y_train)
    lstm = Sequential([Input(shape=(5, X_seq.shape[2])), LSTM(lstm_units), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_seq_train, y_seq_train, epochs=10, verbose=0)
    meta = LinearRegression().fit(np.column_stack((cb.predict(X_train), lstm.predict(X_seq_train, verbose=0).flatten())), y_seq_train)
    preds = meta.predict(np.column_stack((cb.predict(X_test), lstm.predict(X_seq_test, verbose=0).flatten())))
    return mean_squared_error(y_seq_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# === 4. COMPARAÇÃO E RESULTADOS ===
model_lr = LinearRegression().fit(X_train, y_train)
model_bayes = grid_bayes.best_estimator_
model_cb = CatBoostRegressor(iterations=200, verbose=False).fit(X_train, y_train)
model_lstm = Sequential([Input(shape=(5, X_seq.shape[2])), LSTM(study.best_params['lstm_units']), Dense(1)])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_seq_train, y_seq_train, epochs=20, verbose=0)

# Avaliar
avaliar(model_lr.predict(X_test), y_test, "Linear")
avaliar(model_bayes.predict(X_test), y_test, "Bayesian Grid")
avaliar(model_cb.predict(X_test), y_test, "CatBoost")

# === 5. SUGESTÃO FINAL (ENSEMBLE) ===
meta = LinearRegression().fit(np.column_stack((model_cb.predict(X_train), model_lstm.predict(X_seq_train, verbose=0).flatten())), y_seq_train)
p_cb_f = model_cb.predict(df_limpo.iloc[[-1]]).flatten()
p_lstm_f = model_lstm.predict(np.array([df_limpo.iloc[-5:].values]), verbose=0).flatten()
pred_final = meta.predict(np.column_stack((p_cb_f, p_lstm_f)))

print(f"\nSugestão Final (Ensemble + PuLP): {otimizar_com_pulp(pred_final[0])}")