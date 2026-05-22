import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

# =============================================================
# 1. PREPARAÇÃO DOS DADOS HISTÓRICOS (SÉRIES TEMPORAIS)
# =============================================================
print("Preparando dados de séries temporais com alinhamento rigoroso...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

historico_atrasos = []
gabarito_sorteios = [] 
atraso_atual = {d: 0 for d in range(1, 61)}

for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    historico_atrasos.append([atraso_atual[d] for d in range(1, 61)])
    gabarito_sorteios.append(sorteados)
    
    for d in range(1, 61):
        if d in sorteados:
            atraso_atual[d] = 0
        else:
            atraso_atual[d] += 1

matriz_atrasos = np.array(historico_atrasos)
scaler = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler.fit_transform(matriz_atrasos)

window_size = 10

# Alinhamento perfeito de janelas e gabaritos
X_all, y_all, gabarito_all = [], [], []
for i in range(len(dados_norm) - window_size):
    X_all.append(dados_norm[i:(i + window_size), :])
    y_all.append(dados_norm[i + window_size, :])
    gabarito_all.append(gabarito_sorteios[i + window_size])

X_all = np.array(X_all)
y_all = np.array(y_all)

# Separação clássica: 80% treino e 20% teste oculto
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]
gabarito_teste = gabarito_all[train_size:]

# =============================================================
# 2. TREINAMENTO DA REDE NEURAL (LSTM MULTIVARIÁVEL)
# =============================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, 60)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(60, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mean_squared_error')

print("Treinando a rede neural LSTM (Matriz de Ausência)... Por favor, aguarde.")
model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=0)


# =============================================================
# 3. AVALIAÇÃO DE ACURÁCIA (BACKTESTING COMPLETO EM TODO O TESTE)
# =============================================================
print(f"\nAvaliando acurácia histórica em TODOS os {len(X_test)} concursos da base de teste...")
previsoes_test_norm = model.predict(X_test, verbose=0)
previsoes_test_reais = scaler.inverse_transform(previsoes_test_norm)

total_ternos_ganhos = 0
total_quadras_ganhas = 0
total_quinas_ganhas = 0
total_senas_ganhas = 0
total_bilhetes_validados_gerados = 0
dezenas_certas_no_top25 = []

# Varre a base inteira de teste (sem o limite de 50)
for i in range(len(X_test)):
    previsao_concurso = previsoes_test_reais[i]
    resultado_real = gabarito_teste[i]
    
    probabilidades = []
    for idx, atr_prev in enumerate(previsao_concurso):
        score = 1 / (1 + atr_prev)
        probabilidades.append((idx + 1, score))
        
    dezenas_sugeridas = sorted(probabilidades, key=lambda x: x[1], reverse=True)
    top_25_ia = [item[0] for item in dezenas_sugeridas[:25]]
    
    # Avaliação 1: Quantos números do sorteio real entraram no grupo selecionado pela IA
    acertos_puros = len(set(top_25_ia) & resultado_real)
    dezenas_certas_no_top25.append(acertos_puros)
    
    # Avaliação 2: Simular bilhetes reais com o teu validador estrito
    jogos_simulados = []
    tentativas_loop = 0
    # Tenta achar até 5 jogos ideais, se demorar muito passa à frente para não travar
    while len(jogos_simulados) < 5 and tentativas_loop < 1000:
        tentativas_loop += 1
        palpite = sorted(random.sample(top_25_ia, 6))
        if jogo_e_valido_estritamente(palpite):
            jogos_simulados.append(palpite)
            
    for jogo in jogos_simulados:
        total_bilhetes_validados_gerados += 1
        acertos = len(set(jogo) & resultado_real)
        if acertos == 6: total_senas_ganhas += 1
        elif acertos == 5: total_quinas_ganhas += 1
        elif acertos == 4: total_quadras_ganhas += 1
        elif acertos == 3: total_ternos_ganhos += 1

print("-" * 75)
print(f"RELATÓRIO DEFINITIVO DE ACURÁCIA (Total de Teste Oculto):")
print(f"▶ Concursos totais avaliados no passado: {len(X_test)}")
print(f"▶ Total de bilhetes válidos gerados e testados: {total_bilhetes_validados_gerados}")
print(f"▶ Média de dezenas do sorteio capturadas no TOP 25 da IA: {np.mean(dezenas_certas_no_top25):.2f} de 6 possíveis")
print("-" * 75)
print("PREMIAÇÕES REAIS DETECTADAS NO BACKTESTING:")
print(f"▶ TERNOS GERADOS:  {total_ternos_ganhos}")
print(f"▶ QUADRAS GERADAS: {total_quadras_ganhas}")
print(f"▶ QUINAS GERADAS:  {total_quinas_ganhas}")
print(f"▶ SENAS GERADAS:   {total_senas_ganhas}")
print("-" * 75)


# =============================================================
# 4. INTELIGÊNCIA PREDITIVA PARA O PRÓXIMO CONCURSO
# =============================================================
print("\nGerando previsões térmicas globais para o PRÓXIMO concurso...")
ultima_janela = dados_norm[-window_size:]
ultima_janela_reshape = np.reshape(ultima_janela, (1, window_size, 60))

previsao_futura_norm = model.predict(ultima_janela_reshape, verbose=0)
previsao_futura_reais = scaler.inverse_transform(previsao_futura_norm)[0]

probabilidades_futuras = []
for idx, atraso_previsto in enumerate(previsao_futura_reais):
    score_proximidade_zero = 1 / (1 + atraso_previsto) 
    probabilidades_futuras.append((idx + 1, score_proximidade_zero))

dezenas_sugeridas_futuras = sorted(probabilidades_futuras, key=lambda x: x[1], reverse=True)
top_25_atual_ia = [item[0] for item in dezenas_sugeridas_futuras[:25]]

print(f"-> Grupo de Elite Final selecionado pela IA (TOP 25): {sorted(top_25_atual_ia)}")


# =============================================================
# 5. GERAÇÃO DOS 5 JOGOS DEFINITIVOS COM VALIDAÇÃO E ÍNDICES
# =============================================================
print("\nA IA está montando as combinações e validando no seu gerador...")
N_JOGOS = 5
jogos_finais_concurso = []
tentativas_da_rede = 0

while len(jogos_finais_concurso) < N_JOGOS:
    tentativas_da_rede += 1
    palpite_ia = sorted(random.sample(top_25_atual_ia, 6))
    
    if jogo_e_valido_estritamente(palpite_ia):
        jogos_finais_concurso.append(palpite_ia)

# Exibição do Volante Final
print("\n" + "=" * 75)
print("        JOGOS GERADOS EXCLUSIVAMENTE PELA REDE NEURAL LSTM")
print("=" * 75)
print(f"Combinações filtradas na memória até aprovação da lógica: {tentativas_da_rede}")
print("-" * 75)

for i, jogo in enumerate(jogos_finais_concurso, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    
    linha_pandas = pd.Series(
        data=jogo, 
        index=['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    )
    
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice: {indice_br}")

print("=" * 75)
print("Processo completo executado sobre toda a base histórica!")
print("=" * 75)