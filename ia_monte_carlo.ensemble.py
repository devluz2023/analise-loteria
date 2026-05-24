import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. PREPARAÇÃO DOS DADOS E TREINAMENTO DO AUTOENCODER
# =============================================================
print("Carregando dados e inicializando o Autoencoder Profundo...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

matriz_jejuns_bruta = df_ausencia[colunas_jejum].values.astype(np.float32)
scaler_ae = MinMaxScaler(feature_range=(0, 1))
matriz_jejuns_norm = scaler_ae.fit_transform(matriz_jejuns_bruta)

dim_entrada = matriz_jejuns_norm.shape[1]
dim_latente = 16

input_layer = Input(shape=(dim_entrada,))
encoded = Dense(32, activation='relu')(input_layer)
bottleneck = Dense(dim_latente, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(bottleneck)
output_layer = Dense(dim_entrada, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, bottleneck)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(matriz_jejuns_norm, matriz_jejuns_norm, epochs=35, batch_size=32, verbose=0)

# Probabilidades a priori de Bayes
distribuicao_global_jejuns = {int(col): int(df_ausencia[col].sum()) for col in colunas_jejum}
total_sequencias = sum(distribuicao_global_jejuns.values())
prob_prior_jejum = {t: (qtd / total_sequencias) for t, qtd in distribuicao_global_jejuns.items()}

# Matriz temporal para a LSTM
historico_atrasos = []
gabarito_sorteios = [] 
atraso_atual = {d: 0 for d in range(1, 61)}

for index, row in df_cronologico.iterrows():
    sorteados = set([int(row[c]) for c in colunas_bolas])
    historico_atrasos.append([atraso_atual[d] for d in range(1, 61)])
    gabarito_sorteios.append(sorteados)
    for d in range(1, 61):
        if d in sorteados: atraso_atual[d] = 0
        else: atraso_atual[d] += 1

matriz_atrasos = np.array(historico_atrasos)
scaler_lstm = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler_lstm.fit_transform(matriz_atrasos)


# =============================================================
# 2. MOTOR DE SIMULAÇÃO DE MONTE CARLO (MCMC ENGINE)
# =============================================================
def simular_monte_carlo_proximo_concurso(scores_dezenas, atrasos_atuais, n_simulacoes=5000):
    """
    Simula milhares de cenários possíveis para o sorteio usando amostragem
    estocástica baseada nas notas da IA e validando nas restrições de Bayes.
    """
    dezenas = list(range(1, 61))
    scores_array = np.array(scores_dezenas)
    
    # Aplica Softmax para transformar os scores em uma distribuição de probabilidade estrita para a roleta
    exp_scores = np.exp(scores_array / np.max(scores_array)) # Temperatura de amostragem
    probs_roleta = exp_scores / np.sum(exp_scores)
    
    # Dicionário para computar a frequência de aparição de cada jogo válido simulado
    historico_cenarios_validos = {}
    
    for _ in range(n_simulacoes):
        # Monte Carlo: Sorteia um cenário candidado com base nas probabilidades da IA
        cenario_palpite = sorted(np.random.choice(dezenas, size=6, replace=False, p=probs_roleta))
        
        # Filtro de Rejeição de Monte Carlo: O cenário só é aceito se passar no seu gerador estrito
        if jogo_e_valido_estritamente(cenario_palpite):
            # Filtro adicional de Bayes: Evitar acumular dezenas extremamente frias no mesmo cenário
            frias = sum(1 for d in cenario_palpite if atrasos_atuais[d-1] > 18)
            if frias <= 2:
                tuple_jogo = tuple(cenario_palpite)
                if tuple_jogo in historico_cenarios_validos:
                    historico_cenarios_validos[tuple_jogo] += 1
                else:
                    historico_cenarios_validos[tuple_jogo] = 1
                    
    # Ordena as combinações que mais vezes convergiram e sobreviveram em todos os universos simulados
    jogos_mais_frequentes = sorted(historico_cenarios_validos.items(), key=lambda x: x[1], reverse=True)
    
    # Retorna o top 5 de jogos mais estáveis encontrados por Monte Carlo
    return [list(item[0]) for item in jogos_mais_frequentes[:5]]


# =============================================================
# 3. FUNÇÃO OBJETIVO DO OPTUNA
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    peso_lstm = trial.suggest_float('peso_lstm', 0.35, 0.65)
    peso_bayes = 1.0 - peso_lstm
    n_simulacoes_teste = 2000 # Menos simulações nos trials para rodar rápido
    
    X_all, y_all, gabarito_all = [], [], []
    for i in range(len(dados_norm) - window_size):
        X_all.append(dados_norm[i:(i + window_size), :])
        y_all.append(dados_norm[i + window_size, :])
        gabarito_all.append(gabarito_sorteios[i + window_size])

    X_all, y_all = np.array(X_all), np.array(y_all)
    train_size = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train = y_all[:train_size]
    gabarito_teste = gabarito_all[train_size:]
    
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window_size, 60)),
        Dropout(0.2),
        Dense(60, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=8, verbose=0)
    
    previsoes_norm = model.predict(X_test[-50:], verbose=0)
    previsoes_lstm = scaler_lstm.inverse_transform(previsoes_norm)
    gabarito_recortado = gabarito_teste[-50:]
    atrasos_no_teste = matriz_atrasos[train_size + window_size - 1 : -1][-50:]
    
    score_fim = 0
    for i in range(50):
        pred_lstm = previsoes_lstm[i]
        atraso_concurso_atual = atrasos_no_teste[i]
        resultado_real = gabarito_recortado[i]
        
        scores_dezenas = []
        for idx in range(60):
            score_lstm_norm = 1 / (1 + pred_lstm[idx])
            jejum_futuro = atraso_concurso_atual[idx] + 1
            score_bayes_norm = prob_prior_jejum.get(jejum_futuro, 0.0001)
            scores_dezenas.append((peso_lstm * score_lstm_norm) + (peso_bayes * score_bayes_norm))
            
        jogos_simulados = simular_monte_carlo_proximo_concurso(scores_dezenas, atraso_concurso_atual, n_simulacoes=n_simulacoes_teste)
                
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000   # Foco absoluto nas Quadras
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. PROCESSAMENTO MESTRE (MONTE CARLO EXECUTION)
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna para encontrar pesos de equilíbrio da simulação...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros temporais localizados: {params}")
    
    w_size = params['window_size']
    X_all, y_all, gabarito_all = [], [], []
    for i in range(len(dados_norm) - w_size):
        X_all.append(dados_norm[i:(i + w_size), :])
        y_all.append(dados_norm[i + w_size, :]) 
        gabarito_all.append(gabarito_sorteios[i + w_size])

    X_all, y_all = np.array(X_all), np.array(y_all)
    train_size = int(len(X_all) * 0.8)
    
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train = y_all[:train_size]
    gabarito_teste = gabarito_all[train_size:]
    atrasos_no_teste_completo = matriz_atrasos[train_size + w_size - 1 : -1]
    
    print("\n[FASE 2] Treinando Rede Neural LSTM do Ensemble Principal...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    # =============================================================
    # 5. BACKTESTING TOTAL VIA MONTE CARLO MASSIVO
    # =============================================================
    print(f"\n[FASE 3] Rodando Backtesting completo em {len(X_test)} concursos via Universos Paralelos...")
    previsoes_test_norm = model_master.predict(X_test, verbose=0)
    previsoes_test_reais = scaler_lstm.inverse_transform(previsoes_test_norm)
    
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_gerados = 0
    p_lstm = params['peso_lstm']
    p_bayes = 1.0 - p_lstm
    
    for i in range(len(X_test)):
        pred_lstm = previsoes_test_reais[i]
        atraso_concurso_atual = atrasos_no_teste_completo[i]
        resultado_real = gabarito_teste[i]
        
        scores_dezenas = []
        for idx in range(60):
            score_lstm = 1 / (1 + pred_lstm[idx])
            jejum_futuro = atraso_concurso_atual[idx] + 1
            score_bayes = prob_prior_jejum.get(jejum_futuro, 0.0001)
            scores_dezenas.append((p_lstm * score_lstm) + (p_bayes * score_bayes))
            
        # Executa Monte Carlo histórico com 4000 simulações por rodada do passado
        jogos_simulados = simular_monte_carlo_proximo_concurso(scores_dezenas, atraso_concurso_atual, n_simulacoes=4000)
                
        for jogo in jogos_simulados:
            total_bilhetes_gerados += 1
            acertos = len(set(jogo) & resultado_real)
            if acertos == 6: total_senas += 1
            elif acertos == 5: total_quinas += 1
            elif acertos == 4: total_quadras += 1
            elif acertos == 3: total_ternos += 1
            
    taxa_ternos = (total_ternos / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0
    taxa_quadras = (total_quadras / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0

    print("-" * 75)
    print(f"RELATÓRIO DE PERFORMANCE DA SIMULAÇÃO DE MONTE CARLO:")
    print(f"▶ Total de bilhetes sobreviventes de maior convergência: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. PALPITE REAL VIA MONTE CARLO PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 4] Rodando 15.000 Simulações de Monte Carlo para o PRÓXIMO concurso real...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler_lstm.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    scores_futuros = []
    for idx in range(60):
        score_lstm = 1 / (1 + pred_futura_lstm[idx])
        jejum_futuro = atraso_hoje_real[idx] + 1
        scores_futuros.append((p_lstm * score_lstm) + (p_bayes * score_bayes))
        
    # No palpite real, abrimos as comportas: 15 mil sorteios estocásticos para mapear a estabilidade máxima
    jogos_finais = simular_monte_carlo_proximo_concurso(scores_futuros, atraso_hoje_real, n_simulacoes=15000)
            
    print("\n" + "=" * 75)
    print("      JOGOS DEFINITIVOS OBTIDOS POR CONVERGÊNCIA DE MONTE CARLO")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)