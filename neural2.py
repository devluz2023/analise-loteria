import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

# Abafa logs excessivos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. PREPARAÇÃO DOS DADOS HISTÓRICOS E DA CURVA DE BAYES
# =============================================================
print("Carregando dados e calculando probabilidades de Bayes...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

# Carrega a tabela para extrair a probabilidade condicional de cada tamanho de jejum
df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

distribuicao_global_jejuns = {int(col): 0 for col in colunas_jejum}
for index, row in df_ausencia.iterrows():
    for col in colunas_jejum:
        distribuicao_global_jejuns[int(col)] += int(row[col])

total_sequencias = sum(distribuicao_global_jejuns.values())

# P(Jejum = t) -> Distribuição de probabilidade a priori do tamanho do jejum
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
scaler = MinMaxScaler(feature_range=(0, 1))
dados_norm = scaler.fit_transform(matriz_atrasos)


# =============================================================
# 2. FUNÇÃO OBJETIVO: OPTUNA TUNANDO O ENSEMBLE COM MIRA NAS QUADRAS
# =============================================================
def objective(trial):
    # Parâmetros da LSTM
    window_size = trial.suggest_int('window_size', 10, 18)
    
    # MODIFICAÇÃO 1: Alargado o range mínimo do Grupo de Elite para dar margem à Quadra
    top_n_ia = trial.suggest_int('top_n_ia', 24, 32)
    
    lstm_1 = trial.suggest_int('lstm_1', 64, 128, step=32)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    # Parâmetro do Ensemble
    peso_lstm = trial.suggest_float('peso_lstm', 0.3, 0.8)
    peso_bayes = 1.0 - peso_lstm
    
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
    
    # Modelo LSTM enxuto para otimização rápida
    model = Sequential([
        LSTM(lstm_1, return_sequences=False, input_shape=(window_size, 60)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=0)
    
    # Avaliação retroativa rápida (últimos 80 concursos do teste)
    previsoes_norm = model.predict(X_test[-80:], verbose=0)
    previsoes_lstm = scaler.inverse_transform(previsoes_norm)
    gabarito_recortado = gabarito_teste[-80:]
    
    atrasos_no_teste = matriz_atrasos[train_size + window_size - 1 : -1][-80:]
    
    score_fim = 0
    for i in range(80):
        pred_lstm = list(previsoes_lstm[i])
        atraso_concurso_atual = atrasos_no_teste[i]
        resultado_real = gabarito_recortado[i]
        
        scores_ensemble = []
        for idx in range(60):
            dezena = idx + 1
            score_lstm_norm = 1 / (1 + pred_lstm[idx])
            jejum_futuro = atraso_concurso_atual[idx] + 1
            score_bayes_norm = prob_prior_jejum.get(jejum_futuro, 0.0001)
            
            score_final = (peso_lstm * score_lstm_norm) + (peso_bayes * score_bayes_norm)
            scores_ensemble.append((dezena, score_final))
            
        dezenas_sugeridas = sorted(scores_ensemble, key=lambda x: x[1], reverse=True)
        top_rodada = [item[0] for item in dezenas_sugeridas[:top_n_ia]]
        
        jogos_simulados = []
        tentativas = 0
        
        # MODIFICAÇÃO 3: Subiu para 400 tentativas para dar mais fôlego ao gerador estrito
        while len(jogos_simulados) < 5 and tentativas < 400:
            tentativas += 1
            palpite = sorted(random.sample(top_rodada, 6))
            if jogo_e_valido_estritamente(palpite):
                jogos_simulados.append(palpite)
                
        # MODIFICAÇÃO 2: Sistema de pontos rebalanceado (Terno zerado para o Optuna focar em Quadras)
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 3: score_fim += 0        # Ignora o Terno para forçar a rede a subir o nível
            elif acertos == 4: score_fim += 1000   # Peso massivo para a Quadra
            elif acertos == 5: score_fim += 50000  # Peso mestre para a Quina
            
    return score_fim


# =============================================================
# 3. EXECUÇÃO DO PROCESSO MASTER (TREINO FINAL + PALPITE)
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna focado em QUADRAS no Ensemble (LSTM + Bayes)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros do Ensemble localizados: {params}")
    
    # Treinamento final com os pesos calibrados pelo Optuna
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
    
    model_master = Sequential([
        LSTM(params['lstm_1'], return_sequences=False, input_shape=(w_size, 60)),
        Dropout(params['dropout']),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    
    # Damos 45 épocas no treino final para consolidar bem os acertos mapeados
    model_master.fit(X_train, y_train, batch_size=32, epochs=45, verbose=0)
    
    # =============================================================
    # 4. BACKTESTING TOTAL E ACURÁCIA DO ENSEMBLE (PERCENTUAL)
    # =============================================================
    print(f"\n[FASE 2] Rodando Backtesting completo em {len(X_test)} concursos...")
    previsoes_test_norm = model_master.predict(X_test, verbose=0)
    previsoes_test_reais = scaler.inverse_transform(previsoes_test_norm)
    
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_gerados = 0
    p_lstm = params['peso_lstm']
    p_bayes = 1.0 - p_lstm
    dezenas_certas_no_grupo = []
    
    for i in range(len(X_test)):
        pred_lstm = previsoes_test_reais[i]
        atraso_concurso_atual = atrasos_no_teste_completo[i]
        resultado_real = gabarito_teste[i]
        
        scores_ensemble = []
        for idx in range(60):
            score_lstm = 1 / (1 + pred_lstm[idx])
            jejum_futuro = atraso_concurso_atual[idx] + 1
            score_bayes = prob_prior_jejum.get(jejum_futuro, 0.0001)
            
            score_final = (p_lstm * score_lstm) + (p_bayes * score_bayes)
            scores_ensemble.append((idx + 1, score_final))
            
        dezenas_sugeridas = sorted(scores_ensemble, key=lambda x: x[1], reverse=True)
        top_ia_rodada = [item[0] for item in dezenas_sugeridas[:params['top_n_ia']]]
        
        dezenas_certas_no_grupo.append(len(set(top_ia_rodada) & resultado_real))
        
        jogos_simulados = []
        tentativas_loop = 0
        while len(jogos_simulados) < 5 and tentativas_loop < 1000:
            tentativas_loop += 1
            palpite = sorted(random.sample(top_ia_rodada, 6))
            if jogo_e_valido_estritamente(palpite): 
                jogos_simulados.append(palpite)
                
        for jogo in jogos_simulados:
            total_bilhetes_gerados += 1
            acertos = len(set(jogo) & resultado_real)
            if acertos == 6: total_senas += 1
            elif acertos == 5: total_quinas += 1
            elif acertos == 4: total_quadras += 1
            elif acertos == 3: total_ternos += 1
            
    # Cálculo das taxas de acerto percentuais reais no Backtesting (Nota: Aqui mostramos os Ternos reais obtidos)
    taxa_ternos = (total_ternos / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0
    taxa_quadras = (total_quadras / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0
    taxa_quinas = (total_quinas / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0
    taxa_senas = (total_senas / total_bilhetes_gerados) * 100 if total_bilhetes_gerados > 0 else 0

    print("-" * 75)
    print(f"RELATÓRIO DE PERFORMANCE DO ENSEMBLE HÍBRIDO:")
    print(f"▶ Concursos totais de teste avaliados: {len(X_test)}")
    print(f"▶ Total de bilhetes válidos gerados e analisados: {total_bilhetes_gerados}")
    print(f"▶ Média de dezenas capturadas no TOP {params['top_n_ia']} da IA: {np.mean(dezenas_certas_no_grupo):.2f} de 6")
    print(f"▶ Divisão de forças encontrada: IA ({p_lstm*100:.1f}%) | Bayes ({p_bayes*100:.1f}%)")
    print("-" * 75)
    print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE):")
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%  (Esperado acaso: 0.4311%)")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%  (Esperado acaso: 0.0428%)")
    print(f"▶ QUINAS:  {total_quinas} acertos ➔ {taxa_quinas:.4f}%  (Esperado acaso: 0.0006%)")
    print(f"▶ SENAS:   {total_senas} acertos ➔ {taxa_senas:.6f}%  (Esperado acaso: 0.000002%)")
    print("-" * 75)
    
    # =============================================================
    # 5. PALPITE PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 3] Gerando palpites definitivos para o PRÓXIMO concurso...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    scores_futuros = []
    for idx in range(60):
        score_lstm = 1 / (1 + pred_futura_lstm[idx])
        jejum_futuro = atraso_hoje_real[idx] + 1
        score_bayes = prob_prior_jejum.get(jejum_futuro, 0.0001)
        
        score_final = (p_lstm * score_lstm) + (p_bayes * score_bayes)
        scores_futuros.append((idx + 1, score_final))
        
    dezenas_sugeridas_futuras = sorted(scores_futuros, key=lambda x: x[1], reverse=True)
    top_atual_ia = [item[0] for item in dezenas_sugeridas_futuras[:params['top_n_ia']]]
    
    print(f"-> Grupo de Elite do Ensemble (TOP {params['top_n_ia']}): {sorted(top_atual_ia)}")
    
    jogos_finais = []
    while len(jogos_finais) < 5:
        palpite_ia = sorted(random.sample(top_atual_ia, 6))
        if jogo_e_valido_estritamente(palpite_ia): 
            jogos_finais.append(palpite_ia)
            
    print("\n" + "=" * 75)
    print("      JOGOS GERADOS VIA ENSEMBLE (LSTM + INFERÊNCIA DE BAYES)")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6'])
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice: {indice_br}")
    print("=" * 75)