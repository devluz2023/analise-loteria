import numpy as np
import pandas as pd
import requests
import io
import math
import os
from collections import Counter
from matriz_frequencia_quina import gerar_matriz_quina

# === CONFIGURAÇÃO ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TOTAL_POSSIVEL = math.comb(80, 5)

# === 1. DOWNLOAD E PREPARAÇÃO ===
print("Baixando dados oficiais e processando...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df = pd.read_excel(io.BytesIO(response.content))
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
df = df.sort_values(by='Concurso').reset_index(drop=True)

# Gera a matriz de estatísticas completa
df_matriz = gerar_matriz_quina(df)
df_matriz.to_excel("matriz_ausencia_quina.xlsx", index=False)

# Prepara mapas estatísticos convertendo tudo para float (trata strings e notação científica)
mapas_estatisticos = {}
for col in df_matriz.columns:
    if col != 'Dezena':
        # Converte a coluna inteira para numérico, forçando erro se não for possível
        valores = pd.to_numeric(df_matriz[col], errors='coerce').fillna(0)
        mapas_estatisticos[col] = dict(zip(df_matriz['Dezena'], valores))

# === 2. FUNÇÕES ESSENCIAIS ===
def calcular_indice(comb_list):
    comb_list = sorted(comb_list)
    index = 0
    start = 1
    for i in range(5):
        for val in range(start, comb_list[i]):
            index += math.comb(80 - val, 5 - 1 - i)
        start = comb_list[i] + 1
    return index

def calcular_distancia_circular(idx_atual, idx_proximo):
    dist = abs(idx_proximo - idx_atual)
    return min(dist, TOTAL_POSSIVEL - dist)

def maior_sequencia_circular(comb_list):
    comb = sorted([int(n) for n in comb_list])
    grupos = []
    i = 0
    while i < len(comb):
        g = [comb[i]]
        while i + 1 < len(comb) and comb[i+1] == comb[i] + 1:
            g.append(comb[i+1])
            i += 1
        grupos.append(g)
        i += 1
    if len(grupos) > 1 and 1 in grupos[0] and 80 in grupos[-1]:
        grupos[0] = grupos[-1] + grupos[0]; grupos.pop()
    return max([len(g) for g in grupos])

# === 3. PROCESSAMENTO E INTEGRAÇÃO ===
df['Indice'] = df.apply(lambda row: calcular_indice([int(row[c]) for c in colunas_bolas]), axis=1)
ultimo_indice = df['Indice'].iloc[-1]

# Mapeamento de Elite
historico = [tuple(sorted([int(row[col]) for col in colunas_bolas])) for _, row in df.iterrows()]
freq_map = Counter(historico)
df_elite = pd.DataFrame([{'Indice': calcular_indice(c), 'Combinacao': c, 'Frequencia': f} for c, f in freq_map.items()])

# Métricas Originais
df_elite['Distancia'] = df_elite['Indice'].apply(lambda x: calcular_distancia_circular(x, ultimo_indice))
df_elite['quantidade_sequencia'] = df_elite['Combinacao'].apply(maior_sequencia_circular)

# Integração DINÂMICA de TODAS as colunas da matriz
for nome_coluna, mapa in mapas_estatisticos.items():
    df_elite[f'{nome_coluna}_Soma'] = df_elite['Combinacao'].apply(lambda c: sum(mapa.get(n, 0) for n in c))
    df_elite[f'{nome_coluna}_Max'] = df_elite['Combinacao'].apply(lambda c: max([mapa.get(n, 0) for n in c], default=0))

# === 4. RESULTADO ===
df_resultado = df_elite[df_elite['Frequencia'] > 0].sort_values(by='Distancia')

print("\n=== DIAGNÓSTICO INTEGRADO COMPLETO ===")
print(df_resultado.head(15).to_string())


# === EXIBIÇÃO FORMATADA COM TODAS AS SOMAS ===

# Identifica automaticamente todas as colunas de soma que criamos
# === DEFINIÇÃO DAS COLUNAS FIXAS ===
# === DEFINIÇÃO DAS COLUNAS ===
colunas_fixas = [
    'Indice', 
    'Combinacao', 
    'Frequencia', 
    'Distancia', 
    'quantidade_sequencia'
]

# Filtra apenas colunas que terminam com '_Soma' E que são numéricas (excluindo as colunas indesejadas)
# Lista de colunas para EXCLUIR explicitamente
colunas_para_remover = [
    'Total_Streaks_Observed_Soma', 
    'Total_Absence_Draws_Soma', 
    'Average_Streak_Length_Soma',
    'Desvio_Soma',
    'Prob_Nao_Sair_Empirica_1_Concurso_Soma',
    'Prob_Nao_Sair_Empirica_247_Concursos_Soma',
    'Máximo Consecutivo_Soma',
    'Total de Sequências (qualquer tamanho)_Soma',
    'Total de Sequências (> 1)_Soma',
    'Máximo de Sorteios Sem Sair_Soma',
    'Ausencia_Actual_Concursos_Soma'
]

# Seleciona todas as colunas '_Soma', mas remove as que você não quer
cols_soma_dinamicas = [
    c for c in df_elite.columns 
    if c.endswith('_Soma') and c not in colunas_para_remover
]

# Junta tudo
colunas_finais = colunas_fixas + cols_soma_dinamicas

# === EXIBIÇÃO ===
print("\n=== DIAGNÓSTICO FILTRADO (SOMENTE ÍNDICES DE JEJUM) ===")
print(df_resultado[colunas_finais].head(15).to_string(index=False))

# Salva apenas o que você pediu
df_resultado[colunas_finais].to_excel("diagnostico_quina_filtrado.xlsx", index=False)





import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. PREPARAÇÃO DOS DADOS (Inputs Numéricos)
# 'cols_soma_dinamicas' são as colunas geradas anteriormente
colunas_para_lstm = ['Indice', 'Frequencia', 'Distancia', 'quantidade_sequencia'] + cols_soma_dinamicas
features = df_resultado[colunas_para_lstm].values

# Normalização (OBRIGATÓRIO para LSTM)
scaler_features = MinMaxScaler()
scaled_features = scaler_features.fit_transform(features)

# O Alvo é o próprio 'Indice' (o próximo na sequência)
scaler_target = MinMaxScaler()
target = scaler_target.fit_transform(df_resultado[['Indice']].values)

# 2. CRIAÇÃO DE JANELAS (Sequências de 10 sorteios para prever o próximo)
def create_sequences(data, target, seq_length=16):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 16
X, y = create_sequences(scaled_features, target, SEQ_LENGTH)

# 3. MODELO LSTM
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) # Saída: O próximo Índice
])

model.compile(optimizer='adam', loss='mse')

# 4. TREINAMENTO
print("\nIniciando treinamento da LSTM para prever o próximo índice...")
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# 5. PREDIÇÃO DO PRÓXIMO ÍNDICE
# Pegamos as últimas 10 sequências disponíveis
ultimos_dados = scaled_features[-SEQ_LENGTH:]
input_pred = ultimos_dados.reshape(1, SEQ_LENGTH, X.shape[2])

pred_scaled = model.predict(input_pred)
proximo_indice = scaler_target.inverse_transform(pred_scaled)

print(f"\n--- PREDIÇÃO ---")
print(f"O próximo índice previsto pela LSTM é: {int(proximo_indice[0][0])}")