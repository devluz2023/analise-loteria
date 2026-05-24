import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÃO DA SUA LÓGICA DE ÍNDICES ===
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. CARREGAMENTO DOS DADOS E TREINAMENTO DO AUTOENCODER
# =============================================================
print("Carregando dados e construindo Autoencoder para a Matriz de Bayes...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

# Carrega a tabela horizontal de ausência completa
df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

# Prepara os dados para o Autoencoder (Instâncias de jejum histórico)
matriz_jejuns_bruta = df_ausencia[colunas_jejum].values.astype(np.float32)
scaler_ae = MinMaxScaler(feature_range=(0, 1))
matriz_jejuns_norm = scaler_ae.fit_transform(matriz_jejuns_bruta)

# Arquitetura do Autoencoder
dim_entrada = matriz_jejuns_norm.shape[1] # Qtd de colunas de jejum (ex: 108)
dim_latente = 12                          # Gargalo: Comprime para 12 dimensões essenciais

input_layer = Input(shape=(dim_entrada,))
encoded = Dense(32, activation='relu')(input_layer)
bottleneck = Dense(dim_latente, activation='relu', name='espaco_latente')(encoded)
decoded = Dense(32, activation='relu')(bottleneck)
output_layer = Dense(dim_entrada, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, bottleneck) # Modelo cortado que extrai os recursos comprimidos

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# Treina o Autoencoder para reconstruir a si mesmo (Self-Supervised)
autoencoder.fit(matriz_jejuns_norm, matriz_jejuns_norm, epochs=40, batch_size=32, verbose=0)

# Extrai os scores latentes e mapeia de volta um vetor de peso para cada tamanho de jejum
recursos_latentes = encoder.predict(matriz_jejuns_norm, verbose=0)
score_peso_jejum = np.mean(recursos_latentes, axis=0) # Perfil latente médio

# Reconstrói a estrutura a priori para o histórico
distribuicao_global_jejuns = {int(col): 0 for col in colunas_jejum}
for col in colunas_jejum:
    distribuicao_global_jejuns[int(col)] = int(df_ausencia[col].sum())

total_sequencias = sum(distribuicao_global_jejuns.values())
prob_prior_jejum = {t: (qtd / total_sequencias) for t, qtd in distribuicao_global_jejuns.items()}

# Reconstrói a matriz temporal para a LSTM
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
# 2. OPTIMIZADOR NÃO LINEAR COM ENGENHARIA LATENTE DO AUTOENCODER
# =============================================================
def resolver_nlp_com_recursos_autoencoder(scores_dezenas, atrasos_atuais, params_nlp):
    """
    Usa SLSQP para otimizar a escolha das dezenas aplicando pesos ajustados
    pelas features compactadas do Autoencoder.
    """
    scores = np.array(scores_dezenas)
    
    def funcao_objetivo(x):
        x_safe = np.clip(x, 1e-10, 1.0)
        # Termo Não Linear: Entropia Ponderada pelo perfil latente do Autoencoder
        # Força o espalhamento de dezenas nas dimensões que o Autoencoder identificou como estruturais
        entropia_latente = -np.sum(x_safe * np.log(x_safe)) * np.mean(score_peso_jejum)
        return -(np.sum(scores * x) + params_nlp['peso_entropia'] * entropia_latente)

    def restricao_soma(x):
        return np.sum(x) - 6.0

    def restricao_frias(x):
        frias_idx = np.where(atrasos_atuais > params_nlp['corte_jejum_longo'])[0]
        return params_nlp['max_dezenas_frias'] - np.sum(x[frias_idx])

    def restricao_quentes(x):
        quentes_idx = np.where(atrasos_atuais <= 4)[0]
        return np.sum(x[quentes_idx]) - params_nlp['min_dezenas_quentes']

    constraints = [
        {'type': 'eq', 'fun': restricao_soma},
        {'type': 'ineq', 'fun': restricao_frias},
        {'type': 'ineq', 'fun': restricao_quentes}
    ]
    
    bounds = [(0.0, 1.0) for _ in range(60)]
    x0 = np.full(60, 0.1)
    
    res = minimize(funcao_objetivo, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 80, 'disp': False})
    
    if res.success:
        return {d+1: max(0.0, val) for d, val in enumerate(res.x)}
    else:
        return {d+1: 0.1 for d in range(60)}


def amostrar_jogos_da_curva_latente(pesos_nlp, n_jogos=5):
    jogos_gerados = []
    dezenas = list(range(1, 61))
    valores_pesos = np.array([pesos_nlp[d] for d in dezenas])
    valores_pesos = np.where(valores_pesos < 1e-4, 1e-5, valores_pesos)
    
    tentativas = 0
    while len(jogos_gerados) < n_jogos and tentativas < 2000:
        tentativas += 1
        probs_roleta = valores_pesos / np.sum(valores_pesos)
        palpite = sorted(np.random.choice(dezenas, size=6, replace=False, p=probs_roleta))
        
        if sum(1 for n in palpite if n % 2 == 0) in [2, 3, 4]:
            if palpite not in jogos_gerados:
                jogos_gerados.append([int(n) for n in palpite])
                
    return jogos_gerados


# =============================================================
# 3. FUNÇÃO OBJETIVO DO OPTUNA
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    peso_lstm = trial.suggest_float('peso_lstm', 0.4, 0.7)
    peso_bayes = 1.0 - peso_lstm
    
    params_nlp = {
        'peso_entropia': trial.suggest_float('peso_entropia', 0.02, 0.20),
        'corte_jejum_longo': trial.suggest_int('corte_jejum_longo', 14, 24),
        'max_dezenas_frias': trial.suggest_float('max_dezenas_frias', 1.0, 2.0),
        'min_dezenas_quentes': trial.suggest_float('min_dezenas_quentes', 2.0, 3.5)
    }
    
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
    
    previsoes_norm = model.predict(X_test[-60:], verbose=0)
    previsoes_lstm = scaler_lstm.inverse_transform(previsoes_norm)
    gabarito_recortado = gabarito_teste[-60:]
    atrasos_no_teste = matriz_atrasos[train_size + window_size - 1 : -1][-60:]
    
    score_fim = 0
    for i in range(60):
        pred_lstm = previsoes_lstm[i]
        atraso_concurso_atual = atrasos_no_teste[i]
        resultado_real = gabarito_recortado[i]
        
        scores_dezenas = []
        for idx in range(60):
            score_lstm_norm = 1 / (1 + pred_lstm[idx])
            jejum_futuro = atraso_concurso_atual[idx] + 1
            score_bayes_norm = prob_prior_jejum.get(jejum_futuro, 0.0001)
            scores_dezenas.append((peso_lstm * score_lstm_norm) + (peso_bayes * score_bayes_norm))
            
        pesos_nlp = resolver_nlp_com_recursos_autoencoder(scores_dezenas, atraso_concurso_atual, params_nlp)
        jogos_simulados = amostrar_jogos_da_curva_latente(pesos_nlp, n_jogos=5)
                
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. EXECUÇÃO MESTRE DO ECOSSISTEMA COMPATÍVEL
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna no Híbrido com Compressão via Autoencoder...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Configurações de Estado Ótimo Localizadas: {params}")
    
    params_nlp_master = {
        'peso_entropia': params['peso_entropia'],
        'corte_jejum_longo': params['corte_jejum_longo'],
        'max_dezenas_frias': params['max_dezenas_frias'],
        'min_dezenas_quentes': params['min_dezenas_quentes']
    }
    
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
    
    print("\n[FASE 2] Treinando Rede Neural LSTM do Escopo Master...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    # =============================================================
    # 5. BACKTESTING TOTAL
    # =============================================================
    print(f"\n[FASE 3] Rodando Backtesting completo em {len(X_test)} concursos...")
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
            
        pesos_nlp_hist = resolver_nlp_com_recursos_autoencoder(scores_dezenas, atraso_concurso_atual, params_nlp_master)
        jogos_simulados = amostrar_jogos_da_curva_latente(pesos_nlp_hist, n_jogos=5)
                
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
    print(f"RELATÓRIO DE CONVERSÃO COM EXTRAÇÃO LATENTE (AUTOENCODER + NLP):")
    print(f"▶ Total de bilhetes gerados no teste: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. PALPITE REAL PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 4] Gerando palpite definitivo com representação latente profunda...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler_lstm.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    scores_futuros = []
    for idx in range(60):
        score_lstm = 1 / (1 + pred_futura_lstm[idx])
        jejum_futuro = atraso_hoje_real[idx] + 1
        scores_futuros.append((p_lstm * score_lstm) + (p_bayes * score_bayes))
        
    pesos_futuros_nlp = resolver_nlp_com_recursos_autoencoder(scores_futuros, atraso_hoje_real, params_nlp_master)
    jogos_finais = amostrar_jogos_da_curva_latente(pesos_futuros_nlp, n_jogos=5)
            
    print("\n" + "=" * 75)
    print("      JOGOS GERADOS VIA EXTRAÇÃO LATENTE PROFUNDA + ENSEMBLE NLP")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)