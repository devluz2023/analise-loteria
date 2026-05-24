import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÃO DA SUA LÓGICA DE ÍNDICES ===
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. PREPARAÇÃO DOS DADOS HISTÓRICOS E DA CURVA DE BAYES
# =============================================================
print("Carregando dados e preparando ambiente para Programação Linear...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

distribuicao_global_jejuns = {int(col): 0 for col in colunas_jejum}
for index, row in df_ausencia.iterrows():
    for col in colunas_jejum:
        distribuicao_global_jejuns[int(col)] += int(row[col])

total_sequencias = sum(distribuicao_global_jejuns.values())
prob_prior_jejum = {t: (qtd / total_sequencias) for t, qtd in distribuicao_global_jejuns.items()}

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
scaler = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler.fit_transform(matriz_atrasos)


# =============================================================
# 2. OPTIMIZADOR LINEAR PURO (CONTINUOUS PROGRAMMING)
# =============================================================
def resolver_problema_linear_puro(scores_dezenas, atrasos_atuais, config_po):
    """
    Resolve a distribuição de pesos usando Programação Linear Contínua (Simplex).
    Retorna o vetor de pesos ótimos fracionários para cada dezena.
    """
    probs_dezenas_atuais = {
        d: prob_prior_jejum.get(atrasos_atuais[d-1] + 1, 0.0001) for d in range(1, 61)
    }

    prob = LpProblem("Maximizacao_Linear_Pura_Simplex", LpMaximize)
    
    # VARIÁVEIS CONTÍNUAS: Substitui as binárias (MIP) por floats entre 0.0 e 1.0
    x = LpVariable.dicts("peso_dezena", range(1, 61), lowBound=0.0, upBound=1.0, cat="Continuous")
    
    # Função Objetivo: Maximizar o valor esperado do bilhete contínuo
    prob += lpSum(scores_dezenas[d-1] * x[d] for d in range(1, 61))
    
    # --- RESTRIÇÕES LINEARES DA MATRIZ ---
    # A soma dos pesos distribuídos na matriz de 60 dezenas deve ser exatamente igual a 6
    prob += lpSum(x[d] for d in range(1, 61)) == 6.0
    
    # Restrição de dezenas frias e quentes operando sob frações contínuas
    prob += lpSum(x[d] for d in range(1, 61) if atrasos_atuais[d-1] > config_po['corte_jejum_longo']) <= config_po['max_dezenas_frias']
    prob += lpSum(x[d] for d in range(1, 61) if atrasos_atuais[d-1] <= 4) >= config_po['min_dezenas_quentes']
    
    # Janela de envelope de Bayes contínuo
    prob += lpSum(probs_dezenas_atuais[d] * x[d] for d in range(1, 61)) >= config_po['min_soma_bayes']
    prob += lpSum(probs_dezenas_atuais[d] * x[d] for d in range(1, 61)) <= config_po['max_soma_bayes']
    
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Se o simplex convergir para o ótimo, extrai o dicionário de pesos
    if LpStatus[prob.status] == "Optimal":
        return {d: max(0.0, x[d].varValue) for d in range(1, 61)}
    else:
        # Fallback caso os limites de Bayes fiquem travados demais
        return {d: 1.0 if d in sorted(random.sample(range(1, 61), 15)) else 0.0 for d in range(1, 61)}


def amostrar_jogos_da_distribuicao_linear(pesos_continuos, n_jogos=5):
    """
    Usa amostragem probabilística ponderada para converter a solução fracionária
    do Simplex em combinações reais de 6 números.
    """
    jogos_gerados = []
    dezenas = list(range(1, 61))
    
    # Normaliza e adiciona um ruído suave para garantir diversidade entre os bilhetes
    valores_pesos = np.array([pesos_continuos[d] for d in dezenas])
    valores_pesos = np.where(valores_pesos < 0.01, 0.0001, valores_pesos) # Evita probabilidade zero absoluta
    
    tentativas = 0
    while len(jogos_gerados) < n_jogos and tentativas < 2000:
        tentativas += 1
        # Sorteia 6 dezenas usando os pesos do solver contínuo como pesos de probabilidade
        probs_roleta = valores_pesos / np.sum(valores_pesos)
        palpite = sorted(np.random.choice(dezenas, size=6, replace=False, p=probs_roleta))
        
        # Filtros básicos de higienização geométrica pós-amostragem
        if sum(1 for n in palpite if n % 2 == 0) in [2, 3, 4]:
            if palpite not in jogos_gerados:
                jogos_gerados.append([int(n) for n in palpite])
                
    return jogos_gerados


# =============================================================
# 3. OPTUNA REGENDO A ORQUESTRA LINEAR
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    peso_lstm = trial.suggest_float('peso_lstm', 0.4, 0.7)
    peso_bayes = 1.0 - peso_lstm
    
    config_po = {
        'corte_jejum_longo': trial.suggest_int('corte_jejum_longo', 12, 22),
        'max_dezenas_frias': trial.suggest_float('max_dezenas_frias', 1.0, 2.5),
        'min_dezenas_quentes': trial.suggest_float('min_dezenas_quentes', 1.5, 3.5),
        'min_soma_bayes': trial.suggest_float('min_soma_bayes', 0.10, 0.22),
        'max_soma_bayes': trial.suggest_float('max_soma_bayes', 0.38, 0.60)
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
    previsoes_lstm = scaler.inverse_transform(previsoes_norm)
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
            
        # O Simplex calcula a distribuição contínua ideal de pesos
        pesos_otimos = resolver_problema_linear_puro(scores_dezenas, atraso_concurso_atual, config_po)
        jogos_simulados = amostrar_jogos_da_distribuicao_linear(pesos_otimos, n_jogos=5)
                
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. PROCESSAMENTO MESTRE DEFINITIVO
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Iniciando otimização via Programação Linear Pura (Algoritmo Simplex)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros do Ecossistema Linear localizados: {params}")
    
    config_po_master = {
        'corte_jejum_longo': params['corte_jejum_longo'],
        'max_dezenas_frias': params['max_dezenas_frias'],
        'min_dezenas_quentes': params['min_dezenas_quentes'],
        'min_soma_bayes': params['min_soma_bayes'],
        'max_soma_bayes': params['max_soma_bayes']
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
    
    print("\n[FASE 2] Treinando Rede Neural LSTM do Ensemble...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    # =============================================================
    # 5. BACKTESTING TOTAL VIA RELAXAMENTO LINEAR DO ECOSSISTEMA
    # =============================================================
    print(f"\n[FASE 3] Rodando Backtesting completo em {len(X_test)} concursos...")
    previsoes_test_norm = model_master.predict(X_test, verbose=0)
    previsoes_test_reais = scaler.inverse_transform(previsoes_test_norm)
    
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
            
        # O Simplex calcula a distribuição ideal para este ponto do passado
        pesos_distribuídos = resolver_problema_linear_puro(scores_dezenas, atraso_concurso_atual, config_po_master)
        jogos_simulados = amostrar_jogos_da_distribuicao_linear(pesos_distribuídos, n_jogos=5)
                
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
    print(f"RELATÓRIO DE PERFORMANCE DO MODELO DE PROGRAMAÇÃO LINEAR PURA:")
    print(f"▶ Total de bilhetes gerados por Amostragem Ponderada Simplex: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. GERANDO JOGOS PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 4] Gerando distribuição ótima contínua para o PRÓXIMO concurso real...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    scores_futuros = []
    for idx in range(60):
        score_lstm = 1 / (1 + pred_futura_lstm[idx])
        jejum_futuro = atraso_hoje_real[idx] + 1
        scores_futuros.append((p_lstm * score_lstm) + (p_bayes * score_bayes))
        
    pesos_futuros_simplex = resolver_problema_linear_puro(scores_futuros, atraso_hoje_real, config_po_master)
    jogos_finais = amostrar_jogos_da_distribuicao_linear(pesos_futuros_simplex, n_jogos=5)
            
    print("\n" + "=" * 75)
    print("      JOGOS GERADOS VIA DISTRIBUIÇÃO ÓTIMA DO ALGORITMO SIMPLEX")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)