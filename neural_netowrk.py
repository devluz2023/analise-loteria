import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from indice import df_mega_sena

# 1. Preparação dos Dados
# Vamos usar a coluna 'Indice_Combinacao' que criamos
dados = df_mega_sena['Indice_Combinacao'].values.reshape(-1, 1)

# Normalização (Redes neurais trabalham melhor com valores entre 0 e 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler.fit_transform(dados)

# Criar janelas de tempo (Lookback)
# Usaremos os últimos 10 concursos para prever o próximo
def criar_janelas(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 10
X, y = criar_janelas(dados_norm, window_size)

# Divisão Treino e Teste
# Usar os 2000 primeiros concursos para treino e o restante para teste
train_size = 2000
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape para o formato do LSTM: [amostras, janelas, recursos]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Dados de treino X_train shape: {X_train.shape}")
print(f"Dados de treino y_train shape: {y_train.shape}")
print(f"Dados de teste X_test shape: {X_test.shape}")
print(f"Dados de teste y_test shape: {y_test.shape}")

# 2. Construção da Rede Neural (LSTM)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 3. Treinamento
print("Treinando a rede neural... isso pode levar um minuto.")
history = model.fit(X_train, y_train, batch_size=32, epochs=20,
                    validation_data=(X_test, y_test), verbose=1)