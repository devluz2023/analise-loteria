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
print("Carregando dados e extraindo restrições da matriz de Bayes...")
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
# 2. SOLVER MIP COM RESTRIÇÕES DA MATRIZ DE PROBABILIDADE
# =============================================================
def solver_gerar_jogos_bayes_constraints(scores_dezenas, atrasos_atuais, config_po, n_jogos=5, jogos_proibidos=None):
    """
    Usa Programação Inteira para construir jogos baseando-se estritamente
    nas restrições probabilísticas da matriz de ausência.
    """
    jogos_otimizados = []
    if jogos_proibidos is None:
        jogos_proibidos = []

    # Mapeamento de probabilidade individual para cada dezena baseada no seu atraso hoje
    probs_dezenas_atuais = {
        d: prob_prior_jejum.get(atrasos_atuais[d-1] + 1, 0.0001) for d in range(1, 61)
    }

    for rodada in range(n_jogos):
        prob = LpProblem(f"Otimizacao_Bayes_MIP_{rodada}", LpMaximize)
        x = LpVariable.dicts("dezena", range(1, 61), cat="Binary")
        
        # Objetivo: Maximizar o score combinado do Ensemble (IA + Bayes)
        prob += lpSum(scores_dezenas[d-1] * x[d] for d in range(1, 61))
        
        # --- RESTRICÕES ESTRUTURAIS BÁSICAS ---
        prob += lpSum(x[d] for d in range(1, 61)) == 6
        prob += lpSum(x[d] for d in range(1, 61) if d % 2 == 0) == 3
        prob += lpSum(x[d] for d in range(1, 61) if d % 2 != 0) == 3
        
        # --- RESTRICÕES DIRETAS DA MATRIZ DE PROBABILIDADE (BAYES) ---
        # 1. Restrição de dezenas "Frias" (Com jejum longo):
        # Obriga o solver a escolher no máximo X dezenas que estejam com atraso maior que o corte estipulado
        prob += lpSum(x[d] for d in range(1, 61) if atrasos_atuais[d-1] > config_po['corte_jejum_longo']) <= config_po['max_dezenas_frias']
        
        # 2. Restrição de dezenas "Quentes" (Com jejum curto):
        # Garante que o jogo tenha pelo menos Y dezenas com alto índice de fechamento de ciclo (atraso curto)
        prob += lpSum(x[d] for d in range(1, 61) if atrasos_atuais[d-1] <= 4) >= config_po['min_dezenas_quentes']
        
        # 3. Janela de Probabilidade Coletiva:
        # A soma das probabilidades de Bayes do bilhete não pode ser absurdamente baixa (jogo impossível) nem alta demais (jogo óbvio)
        prob += lpSum(probs_dezenas_atuais[d] * x[d] for d in range(1, 61)) >= config_po['min_soma_bayes']
        prob += lpSum(probs_dezenas_atuais[d] * x[d] for d in range(1, 61)) <= config_po['max_soma_bayes']

        # 4. Evitar repetição pesada com bilhetes gerados anteriormente nas rodadas
        for jogo_existente in jogos_otimizados + jogos_proibidos:
            prob += lpSum(x[d] for d in jogo_existente) <= 3
            
        # Executa o solver
        prob.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[prob.status] == "Optimal":
            jogo_resolvido = [d for d in range(1, 61) if x[d].varValue == 1]
            jogos_otimizados.append(sorted(jogo_resolvido))
        else:
            # Se o espaço de restrição probabilística for violado, quebra e retorna o que conseguiu
            break
            
    return jogos_otimizados


# =============================================================
# 3. OPTUNA CALIBRANDO OS LIMITES DAS RESTRIÇÕES DA MATRIZ
# =============================================================
def objective(trial):
    window_size = trial.suggest_int('window_size', 10, 16)
    peso_lstm = trial.suggest_float('peso_lstm', 0.4, 0.7)
    peso_bayes = 1.0 - peso_lstm
    
    # O Optuna agora calibra os parâmetros das restrições de Pesquisa Operacional!
    config_po = {
        'corte_jejum_longo': trial.suggest_int('corte_jejum_longo', 12, 20),
        'max_dezenas_frias': trial.suggest_int('max_dezenas_frias', 1, 2),
        'min_dezenas_quentes': trial.suggest_int('min_dezenas_quentes', 2, 4),
        'min_soma_bayes': trial.suggest_float('min_soma_bayes', 0.15, 0.25),
        'max_soma_bayes': trial.suggest_float('max_soma_bayes', 0.35, 0.55)
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
            
        # Roda o solver com as restrições de matriz dinâmicas do trial
        jogos_simulados = solver_gerar_jogos_bayes_constraints(scores_dezenas, atraso_concurso_atual, config_po, n_jogos=5)
                
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000   # Foco total nas Quadras
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 4. EXECUÇÃO MASTER COMPLETA
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Optuna calibrando restrições MIP baseadas na Matriz de Probabilidade...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros estruturais e probabilísticos localizados: {params}")
    
    # Separa as configurações do solver que o Optuna validou como campeãs
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
    
    print("\n[FASE 2] Treinando Rede Neural LSTM Final...")
    model_master = Sequential([
        LSTM(128, return_sequences=False, input_shape=(w_size, 60)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(60, activation='sigmoid')
    ])
    model_master.compile(optimizer='adam', loss='mean_squared_error')
    model_master.fit(X_train, y_train, batch_size=32, epochs=40, verbose=0)
    
    # =============================================================
    # 5. BACKTESTING HISTÓRICO COM RESTRIÇÕES DE MATRIZ ATIVAS
    # =============================================================
    print(f"\n[FASE 3] Executando Backtesting em {len(X_test)} concursos com Filtro de Restrição Operacional...")
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
            
        # O Solver MIP monta os jogos forçando o cumprimento da distribuição da planilha
        jogos_simulados = solver_gerar_jogos_bayes_constraints(scores_dezenas, atraso_concurso_atual, config_po_master, n_jogos=5)
                
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
    print(f"RELATÓRIO DE ACURÁCIA OPERACIONAL (RESTRIÇÕES DA PLANILHA DE BAYES):")
    print(f"▶ Total de bilhetes gerados via MIP no teste: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 6. PALPITE PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 4] Computando restrições da matriz atual para o PRÓXIMO concurso real...")
    ultima_janela = dados_norm[-w_size:]
    ultima_janela_reshape = np.reshape(ultima_janela, (1, w_size, 60))
    
    pred_futura_lstm = scaler.inverse_transform(model_master.predict(ultima_janela_reshape, verbose=0))[0]
    atraso_hoje_real = matriz_atrasos[-1]
    
    scores_futuros = []
    for idx in range(60):
        score_lstm = 1 / (1 + pred_futura_lstm[idx])
        jejum_futuro = atraso_hoje_real[idx] + 1
        scores_futuros.append((p_lstm * score_lstm) + (p_bayes * score_bayes))
        
    jogos_finais = solver_gerar_jogos_bayes_constraints(scores_futuros, atraso_hoje_real, config_po_master, n_jogos=5)
            
    print("\n" + "=" * 75)
    print("      JOGOS GERADOS VIA MIP (SOLVER CBC) + CRITÉRIOS DE RESTRIÇÃO DE BAYES")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br}")
    print("=" * 75)