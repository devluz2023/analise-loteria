import numpy as np
import pandas as pd
import random
import optuna
import os
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

# Abafa logs excessivos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. PREPARAÇÃO DOS DADOS HISTÓRICOS BASEADOS EM ÍNDICES
# =============================================================
print("Carregando dados históricos e convertendo sorteios em ÍNDICES...")
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
prob_prior_jejum = {t: (qtd / total_sequencias) for t, qtd in distribuicao_global_jejuns.items()}

# Criamos a série temporal de atrasos E a lista de índices reais históricos
historico_atrasos = []
lista_indices_reais = []
gabarito_sorteios = []
atraso_atual = {d: 0 for d in range(1, 61)}

for index, row in df_cronologico.iterrows():
    sorteados_lista = sorted([int(row[c]) for c in colunas_bolas])
    sorteados_set = set(sorteados_lista)
    gabarito_sorteios.append(sorteados_set)
    historico_atrasos.append([atraso_atual[d] for d in range(1, 61)])
    
    linha_pandas = pd.Series(data=sorteados_lista, index=colunas_bolas)
    idx_combinacao = encontrar_posicao_itertools(linha_pandas)
    lista_indices_reais.append(idx_combinacao)
    
    for d in range(1, 61):
        if d in sorteados_set: atraso_atual[d] = 0
        else: atraso_atual[d] += 1

matriz_atrasos = np.array(historico_atrasos)
vetor_indices = np.array(lista_indices_reais).reshape(-1, 1)

scaler_atrasos = MinMaxScaler(feature_range=(0, 1))
scaler_indices = MinMaxScaler(feature_range=(0, 1))

atrasos_norm = scaler_atrasos.fit_transform(matriz_atrasos)
indices_norm = scaler_indices.fit_transform(vetor_indices)


# =============================================================
# 2. FUNÇÕES AUXILIARES: DECODIFICAÇÃO E ENGENHARIA DE FEATURES
# =============================================================
def obter_jogo_por_indice(target_index):
    from math import comb
    jogo = []
    n = 60
    k = 6
    index_atual = target_index - 1
    
    for i in range(1, 61):
        combinacoes_restantes = comb(n - i, k - 1)
        if index_atual < combinacoes_restantes:
            jogo.append(i)
            k -= 1
            if k == 0:
                break
        else:
            index_atual -= combinacoes_restantes
    return jogo

def extrair_features_do_indice(jogo, atrasos_atuais):
    """ Extrai a assinatura estocástica que o CatBoost vai avaliar no jogo gerado """
    probs_bayes = [prob_prior_jejum.get(atrasos_atuais[num-1] + 1, 0.0001) for num in jogo]
    return [
        np.mean(probs_bayes),                  # Média de aderência à curva horizontal
        np.std(probs_bayes),                   # Volatilidade Bayes
        sum(1 for num in jogo if num % 2 == 0), # Balanço de pares
        np.mean(jogo),                         # Centro de massa do volante
        np.std(jogo)                           # Dispersão espacial das dezenas
    ]


# =============================================================
# 3. FUNÇÃO OBJETIVO: OPTUNA TUNANDO O SISTEMA HÍBRIDO (ÍNDICES)
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    tolerancia_busca = trial.suggest_int('tolerancia_busca', 10000, 30000)
    corte_catboost = trial.suggest_float('corte_catboost', 0.3, 0.7)
    
    X_all, y_all, gabarito_all = [], [], []
    for i in range(len(indices_norm) - window_size):
        X_all.append(atrasos_norm[i:(i + window_size), :])
        y_all.append(indices_norm[i + window_size, 0])
        gabarito_all.append(gabarito_sorteios[i + window_size])

    X_all, y_all = np.array(X_all), np.array(y_all)
    train_size = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train = y_all[:train_size]
    gabarito_teste = gabarito_all[train_size:]
    
    # 3.1 Treinar Regressor LSTM rápido
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window_size, 60)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=8, verbose=0)
    
    # 3.2 Treinar CatBoost Regressor rápido para avaliar os desvios de pontuação
    X_cb_train, y_cb_train = [], []
    atrasos_treino_cb = matriz_atrasos[window_size:train_size]
    gabarito_treino_cb = gabarito_sorteios[window_size:train_size]
    
    for idx_c in range(0, len(gabarito_treino_cb), 30): 
        real = list(gabarito_treino_cb[idx_c])
        atr = atrasos_treino_cb[idx_c]
        if len(real) == 6:
            X_cb_train.append(extrair_features_do_indice(real, atr))
            y_cb_train.append(1.0) # Alvo ideal
            fake = sorted(random.sample(range(1, 61), 6))
            X_cb_train.append(extrair_features_do_indice(fake, atr))
            y_cb_train.append(0.0) # Alvo ruim
            
    cb_regressor = CatBoostRegressor(iterations=30, depth=4, verbose=0)
    cb_regressor.fit(np.array(X_cb_train), np.array(y_cb_train))
    
    # 3.3 Avaliação retroativa nos últimos 60 concursos do bloco de validação
    previsoes_norm = model.predict(X_test[-60:], verbose=0)
    previsoes_indices = scaler_indices.inverse_transform(previsoes_norm).flatten()
    gabarito_recortado = gabarito_teste[-60:]
    atrasos_no_teste = matriz_atrasos[train_size + window_size - 1 : -1][-60:]
    
    score_fim = 0
    for i in range(60):
        idx_alvo_previsto = int(previsoes_indices[i])
        resultado_real = gabarito_recortado[i]
        atraso_atual_concurso = atrasos_no_teste[i]
        
        jogos_simulados = []
        passo = 1
        while len(jogos_simulados) < 5 and passo <= tolerancia_busca:
            for sinal in [1, -1]:
                idx_candidato = idx_alvo_previsto + (passo * sinal)
                if 1 <= idx_candidato <= 50063860:
                    palpite = obter_jogo_por_indice(idx_candidato)
                    if jogo_e_valido_estritamente(palpite):
                        # Validação pelo CatBoost Regressor
                        feats = np.array([extrair_features_do_indice(palpite, atraso_atual_concurso)])
                        score_qualidade = cb_regressor.predict(feats)[0]
                        
                        if score_qualidade >= corte_catboost:
                            jogos_simulados.append(palpite)
                            if len(jogos_simulados) == 5: break
            passo += 1
            
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. EXECUÇÃO DO TREINAMENTO MASTER DEFINITIVO
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna no Sistema Combinado de Índices...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros estruturais localizados: {params}")
    
    w_size = params['window_size']
    X_all, y_all, gabarito_all = [], [], []
    for i in range(len(indices_norm) - w_size):
        X_all.append(atrasos_norm[i:(i + w_size), :])
        y_all.append(indices_norm[i + w_size, 0])
        gabarito_all.append(gabarito_sorteios[i + w_size])

    X_all, y_all = np.array(X_all), np.array(y_all)
    train_size = int(len(X_all) * 0.8)
    
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train = y_all[:train_size]
    gabarito_teste = gabarito_all[train_size:]
    atrasos_no_teste_completo = matriz_atrasos[train_size + w_size - 1 : -1]
    
    print("\n[FASE 2] Treinando Regressor LSTM Master para os Índices Alvo...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    print("Treinando o Avaliador Estatístico CatBoost Regressor Master...")
    X_cb_master, y_cb_master = [], []
    atrasos_treino_master = matriz_atrasos[w_size:train_size]
    gabarito_treino_master = gabarito_sorteios[w_size:train_size]
    
    for idx_c in range(len(gabarito_treino_master)):
        real = list(gabarito_treino_master[idx_c])
        atr = atrasos_treino_master[idx_c]
        if len(real) == 6:
            X_cb_master.append(extrair_features_do_indice(real, atr))
            y_cb_master.append(1.0)
            
            for _ in range(2): # Amostragem de contraste para treinar a rejeição
                fake = sorted(random.sample(range(1, 61), 6))
                X_cb_master.append(extrair_features_do_indice(fake, atr))
                y_cb_master.append(0.0)

    cb_master = CatBoostRegressor(iterations=120, depth=6, learning_rate=0.08, verbose=0)
    cb_master.fit(np.array(X_cb_master), np.array(y_cb_master))
    
    # =============================================================
    # 5. BACKTESTING TOTAL UTILIZANDO ÍNDICES FILTRADOS
    # =============================================================
    print(f"\n[FASE 3] Executando Backtesting em {len(X_test)} concursos...")
    previsoes_test_norm = model_master.predict(X_test, verbose=0)
    previsoes_test_reais = scaler_indices.inverse_transform(previsoes_test_norm).flatten()
    
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_gerados = 0
    
    for i in range(len(X_test)):
        idx_centro_previsto = int(previsoes_test_reais[i])
        resultado_real = gabarito_teste[i]
        atraso_atual_concurso = atrasos_no_teste_completo[i]
        
        jogos_simulados = []
        passo = 1
        while len(jogos_simulados) < 5 and passo <= params['tolerancia_busca']:
            for sinal in [1, -1]:
                idx_candidato = idx_centro_previsto + (passo * sinal)
                if 1 <= idx_candidato <= 50063860:
                    palpite = obter_jogo_por_indice(idx_candidato)
                    if jogo_e_valido_estritamente(palpite):
                        # O CatBoost filtra os vizinhos ruins do índice central previsto
                        feats = np.array([extrair_features_do_indice(palpite, atraso_atual_concurso)])
                        if cb_master.predict(feats)[0] >= params['corte_catboost']:
                            jogos_simulados.append(palpite)
                            if len(jogos_simulados) == 5: break
            passo += 1
            
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
    print(f"RELATÓRIO DE PERFORMANCE DA IA COGNITIVA BASEADA EM ÍNDICES:")
    print(f"▶ Total de bilhetes válidos validados pelo CatBoost: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. PROJEÇÃO DO PRÓXIMO CONCURSO REAL
    # =============================================================
    print("\n[FASE 4] Projetando e refinando o índice ideal para o PRÓXIMO concurso...")
    ultima_janela = atrasos_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_norm = model_master.predict(ultima_janela_reshape, verbose=0)
    idx_futuro_centro = int(scaler_indices.inverse_transform(pred_futura_norm)[0][0])
    atraso_hoje_real = matriz_atrasos[-1]
    
    print(f"➔ ÍNDICE TARGET CENTRAL PROJETADO PELA LSTM: {idx_futuro_centro:,}".replace(",", "."))
    
    jogos_finais = []
    passo = 1
    while len(jogos_finais) < 5:
        for sinal in [1, -1]:
            idx_candidato = idx_futuro_centro + (passo * sinal)
            if 1 <= idx_candidato <= 50063860:
                palpite_ia = obter_jogo_por_indice(idx_candidato)
                if jogo_e_valido_estritamente(palpite_ia):
                    feats = np.array([extrair_features_do_indice(palpite_ia, atraso_hoje_real)])
                    if cb_master.predict(feats)[0] >= params['corte_catboost']:
                        jogos_finais.append((idx_candidato, palpite_ia))
                        if len(jogos_finais) == 5: break
        passo += 1
            
    print("\n" + "=" * 75)
    print("      JOGOS DEFINITIVOS: PROXIMIDADE DE ÍNDICE + CORREÇÃO CATBOOST")
    print("=" * 75)
    for i, (idx_gerado, jogo) in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        indice_br = f"{idx_gerado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)