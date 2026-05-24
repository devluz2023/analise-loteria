import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. CONSTRUÇÃO COMPLETA DA MATRIZ DE TRANSIÇÃO DE MARKOV
# =============================================================
print("Carregando Excel e calculando a Matriz de Transição de Markov...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [int(col) for col in df_ausencia.columns if str(col).isdigit()]
max_jejum_mapeado = max(colunas_jejum)

# Dicionário de transições de Markov: Estado t -> {0: prob_sair, t+1: prob_continuar_em_jejum}
matriz_transicao_markov = {}

for t in colunas_jejum:
    total_ocorrencias_estado_t = df_ausencia[str(t)].sum()
    
    if total_ocorrencias_estado_t > 0:
        qtd_transicoes_para_zero = total_ocorrencias_estado_t 
        qtd_transicoes_para_proximo = sum(df_ausencia[str(j)].sum() for j in colunas_jejum if j > t)
        
        total_passagens = qtd_transicoes_para_zero + qtd_transicoes_para_proximo
        
        prob_sair = qtd_transicoes_para_zero / total_passagens
        prob_acumular = qtd_transicoes_para_proximo / total_passagens
    else:
        prob_sair = 0.05 
        prob_acumular = 0.95
        
    matriz_transicao_markov[t] = {0: prob_sair, t+1: prob_acumular}

# Reconstrói a série temporal de atrasos para alimentar a LSTM
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
# 2. MOTOR DE SIMULAÇÃO MARKOVIANA (MARKOV CHAIN SIMULATOR)
# =============================================================
def simular_proximo_estado_markov(pred_lstm, atrasos_atuais, n_simulacoes_mcmc=10000):
    """
    Simula o próximo concurso fazendo as Cadeias de Markov de cada dezena 
    caminharem sob a influência das probabilidades da rede neural LSTM.
    """
    dezenas = list(range(1, 61))
    frequencia_combinacoes_validas = {}
    
    prob_markov_quebra = []
    for idx in range(60):
        jejum_hoje = atrasos_atuais[idx]
        p_zero_markov = matriz_transicao_markov.get(jejum_hoje, {0: 0.1})[0]
        prob_markov_quebra.append(p_zero_markov)
        
    prob_markov_quebra = np.array(prob_markov_quebra)
    score_ia_temporal = 1 / (1 + pred_lstm)
    
    propensao_sorteio = prob_markov_quebra * score_ia_temporal
    probs_roleta = propensao_sorteio / np.sum(propensao_sorteio)
    
    for _ in range(n_simulacoes_mcmc):
        cenario_sorteio = sorted(np.random.choice(dezenas, size=6, replace=False, p=probs_roleta))
        
        # CORRIGIDO: Corrigida a chamada da função para bater com o import do seu gerador.py
        if jogo_e_valido_estritamente(cenario_sorteio):
            tuple_jogo = tuple(cenario_sorteio)
            if tuple_jogo in frequencia_combinacoes_validas:
                frequencia_combinacoes_validas[tuple_jogo] += 1
            else:
                frequencia_combinacoes_validas[tuple_jogo] = 1
                
    jogos_ordenados = sorted(frequencia_combinacoes_validas.items(), key=lambda x: x[1], reverse=True)
    return [list(item[0]) for item in jogos_ordenados[:5]]


# =============================================================
# 3. OPTUNA TUNANDO O ECOSSISTEMA MARKOVIANO
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    peso_lstm = trial.suggest_float('peso_lstm', 0.4, 0.7)
    
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
        
        jogos_simulados = simular_proximo_estado_markov(pred_lstm, atraso_concurso_atual, n_simulacoes_mcmc=2500)
                
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000   
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. EXECUÇÃO DO TREINAMENTO MESTRE (MARKOV MASTER)
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna para equilibrar a Cadeia de Transição de Markov...")
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
    
    print("\n[FASE 2] Treinando Rede Neural LSTM do Ensemble de Markov...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    # =============================================================
    # 5. BACKTESTING TOTAL MARKOVIANO
    # =============================================================
    print(f"\n[FASE 3] Executando Backtesting completo em {len(X_test)} concursos via Cadeias de Markov...")
    previsoes_test_norm = model_master.predict(X_test, verbose=0)
    previsoes_test_reais = scaler_lstm.inverse_transform(previsoes_test_norm)
    
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_gerados = 0
    
    for i in range(len(X_test)):
        pred_lstm = previsoes_test_reais[i]
        atraso_concurso_atual = atrasos_no_teste_completo[i]
        resultado_real = gabarito_teste[i]
        
        jogos_simulados = simular_proximo_estado_markov(pred_lstm, atraso_concurso_atual, n_simulacoes_mcmc=5000)
                
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
    print(f"RELATÓRIO DE PERFORMANCE DE CONVERSÃO HISTÓRICA DE MARKOV:")
    print(f"▶ Total de bilhetes gerados por convergência na Cadeia: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. PALPITE REAL VIA CADEIA DE MARKOV PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 4] Rodando Projeção Estacionária de 20.000 Passos de Markov para hoje...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler_lstm.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    jogos_finais = simular_proximo_estado_markov(pred_futura_lstm, atraso_hoje_real, n_simulacoes_mcmc=20000)
            
    print("\n" + "=" * 75)
    print("      JOGOS DEFINITIVOS: CONVERGÊNCIA DE MATRIZ DE TRANSIÇÃO DE MARKOV")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)