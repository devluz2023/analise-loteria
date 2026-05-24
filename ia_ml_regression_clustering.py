import numpy as np
import pandas as pd
import random
import optuna
import os
from collections import Counter 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

# Abafa logs excessivos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================
# 1. PREPARAÇÃO DOS DADOS HISTÓRICOS E EXTRAÇÃO DE FEATURES
# =============================================================
print("Carregando dados e gerando matriz de Features estruturais...")
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

# Criando a tabela de features macro por concurso para o Clustering Não Rotulado
X_macro_features = []
for i, sorteio in enumerate(gabarito_sorteios):
    jogo_lista = sorted(list(sorteio))
    atrasos_no_concurso = matriz_atrasos[i]
    
    if len(jogo_lista) == 6:
        probs_bayes = [prob_prior_jejum.get(atrasos_no_concurso[num-1] + 1, 0.0001) for num in jogo_lista]
        
        features = [
            sum(jogo_lista),                                      # Soma das dezenas
            sum(1 for n in jogo_lista if n % 2 == 0),            # Qtd de pares
            np.mean(probs_bayes),                                 # Média de Bayes do sorteio
            sum(atrasos_no_concurso[num-1] for num in jogo_lista) # Soma total dos atrasos
        ]
        X_macro_features.append(features)

X_macro_features = np.array(X_macro_features)
scaler_features = MinMaxScaler()
X_macro_norm = scaler_features.fit_transform(X_macro_features)


# =============================================================
# 2. FUNÇÃO OBJETIVO DO OPTUNA: UNINDO CLUSTERING + REGRESSÃO
# =============================================================
def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 3, 7)
    window_size = trial.suggest_int('window_size', 8, 15)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_concursos = kmeans.fit_predict(X_macro_norm)
    
    X_reg, y_reg, gabarito_reg = [], [], []
    for i in range(len(X_macro_norm) - window_size):
        X_reg.append(X_macro_norm[i:(i + window_size), :].flatten())
        y_reg.append(X_macro_norm[i + window_size, :]) 
        gabarito_reg.append(gabarito_sorteios[i + window_size])
        
    X_reg, y_reg = np.array(X_reg), np.array(y_reg)
    train_size = int(len(X_reg) * 0.8)
    
    X_train, X_test = X_reg[:train_size], X_reg[train_size:]
    y_train = y_reg[:train_size]
    gabarito_teste = gabarito_reg[train_size:]
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    previsoes_features_norm = regressor.predict(X_test[-60:])
    previsoes_reais = scaler_features.inverse_transform(previsoes_features_norm)
    gabarito_recortado = gabarito_teste[-60:]
    
    score_fim = 0
    for i in range(min(60, len(previsoes_reais))):
        assinatura_prevista = previsoes_reais[i]
        resultado_real = gabarito_recortado[i]
        
        cluster_atual = labels_concursos[train_size + i]
        indices_do_cluster = np.where(labels_concursos == cluster_atual)[0]
        
        dezenas_do_cluster = []
        for idx_c in indices_do_cluster[-150:]: 
            if idx_c < len(gabarito_sorteios):
                dezenas_do_cluster.extend(list(gabarito_sorteios[idx_c]))
                
        contagem_dezenas = Counter(dezenas_do_cluster)
        top_dezenas_cluster = [item[0] for item in contagem_dezenas.most_common(26)]
        
        jogos_simulados = []
        tentativas = 0
        soma_target = assinatura_prevista[0]
        pares_target = int(np.clip(round(assinatura_prevista[1]), 2, 4))
        tolerancia_soma_trial = 22
        
        while len(jogos_simulados) < 5 and tentativas < 400:
            tentativas += 1
            palpite = sorted(random.sample(top_dezenas_cluster, 6))
            
            if tentativas > 250:
                tolerancia_soma_trial = 35
                
            if abs(sum(palpite) - soma_target) <= tolerancia_soma_trial:
                if tentativas > 300 or sum(1 for n in palpite if n % 2 == 0) == pares_target:
                    if jogo_e_valido_estritamente(palpite):
                        jogos_simulados.append(palpite)
                        
        for jogo in jogos_simulados:
            acertos = len(set(jogo) & resultado_real)
            if acertos == 4: score_fim += 1000
            elif acertos == 5: score_fim += 50000
            
    return score_fim


# =============================================================
# 3. TREINAMENTO MESTRE COM PARÂMETROS CAMPEÕES
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Rodando Optuna no pipeline Combinado ML (K-Means + Regressão)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    params = study.best_params
    print(f"\n-> Melhores parâmetros de ML localizados: {params}")
    
    kmeans_master = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
    labels_concursos_master = kmeans_master.fit_predict(X_macro_norm)
    
    w_size = params['window_size']
    X_reg, y_reg, gabarito_reg = [], [], []
    for i in range(len(X_macro_norm) - w_size):
        X_reg.append(X_macro_norm[i:(i + w_size), :].flatten())
        y_reg.append(X_macro_norm[i + w_size, :])
        gabarito_reg.append(gabarito_sorteios[i + w_size])
        
    X_reg, y_reg = np.array(X_reg), np.array(y_reg)
    train_size = int(len(X_reg) * 0.8)
    
    X_train, X_test = X_reg[:train_size], X_reg[train_size:]
    y_train = y_reg[:train_size]
    gabarito_teste = gabarito_reg[train_size:]
    
    regressor_master = LinearRegression()
    regressor_master.fit(X_train, y_train)
    
    # =============================================================
    # 4. BACKTESTING HISTÓRICO INTEGRADO
    # =============================================================
    print(f"\n[FASE 2] Executando Backtesting completo em {len(X_test)} concursos...")
    previsoes_test_norm = regressor_master.predict(X_test)
    previsoes_test_reais = scaler_features.inverse_transform(previsoes_test_norm)
    
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_gerados = 0
    
    for i in range(len(X_test)):
        assinatura_prevista = {
            'soma': previsoes_test_reais[i][0],
            'pares': int(np.clip(round(previsoes_test_reais[i][1]), 2, 4))
        }
        resultado_real = gabarito_teste[i]
        
        cluster_atual = labels_concursos_master[train_size + i]
        indices_do_cluster = np.where(labels_concursos_master == cluster_atual)[0]
        
        dezenas_do_cluster = []
        for idx_c in indices_do_cluster:
            if idx_c < (train_size + i):
                dezenas_do_cluster.extend(list(gabarito_sorteios[idx_c]))
                
        contagem_dezenas = Counter(dezenas_do_cluster)
        top_dezenas_cluster = [item[0] for item in contagem_dezenas.most_common(26)]
        
        jogos_simulados = []
        tentativas_loop = 0
        tolerancia_soma_backtest = 25
        
        while len(jogos_simulados) < 5 and tentativas_loop < 600:
            tentativas_loop += 1
            palpite = sorted(random.sample(top_dezenas_cluster, 6))
            
            if tentativas_loop > 400:
                tolerancia_soma_backtest = 40
                
            if abs(sum(palpite) - assinatura_prevista['soma']) <= tolerancia_soma_backtest:
                if tentativas_loop > 500 or sum(1 for n in palpite if n % 2 == 0) == assinatura_prevista['pares']:
                    if jogo_e_valido_estritamente(palpite):
                        jogos_simulados.append(palpite)
                        
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
    print(f"RELATÓRIO DE ACURÁCIA DO PIPELINE ML CLÁSSICO:")
    print(f"▶ Total de bilhetes gerados no teste por consistência de Cluster: {total_bilhetes_gerados}")
    print("-" * 75)
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 75)
    
    # =============================================================
    # 5. GERANDO JOGOS PARA O PRÓXIMO CONCURSO
    # =============================================================
    print("\n[FASE 3] Projetando parâmetros e gerando jogos para o PRÓXIMO concurso real...")
    ultima_janela = X_macro_norm[-w_size:]
    ultima_janela_flatten = ultima_janela.flatten().reshape(1, -1)
    
    pred_futura_norm = regressor_master.predict(ultima_janela_flatten)
    pred_futura_reais = scaler_features.inverse_transform(pred_futura_norm)[0]
    
    soma_futura = pred_futura_reais[0]
    pares_futuros = int(np.clip(round(pred_futura_reais[1]), 2, 4))
    print(f"➔ ASSINATURA LINEAR PREVISTA: Soma Target: {int(soma_futura)} | Pares Esperados: {pares_futuros}")
    
    proximo_cluster = kmeans_master.predict(pred_futura_norm)[0]
    indices_do_cluster = np.where(labels_concursos_master == proximo_cluster)[0]
    
    dezenas_do_cluster = []
    for idx_c in indices_do_cluster:
        dezenas_do_cluster.extend(list(gabarito_sorteios[idx_c]))
    contagem_dezenas = Counter(dezenas_do_cluster)
    top_dezenas_cluster_atual = [item[0] for item in contagem_dezenas.most_common(26)]
    
    jogos_finais = []
    tentativas_finais = 0
    tolerancia_soma = 22
    
    while len(jogos_finais) < 5 and tentativas_finais < 4000:
        tentativas_finais += 1
        palpite_ia = sorted(random.sample(top_dezenas_cluster_atual, 6))
        
        if tentativas_finais % 800 == 0:
            tolerancia_soma += 6
            
        if abs(sum(palpite_ia) - soma_futura) <= tolerancia_soma:
            condicao_paridade = (sum(1 for n in palpite_ia if n % 2 == 0) == pares_futuros) if tentativas_finais < 1500 else (sum(1 for n in palpite_ia if n % 2 == 0) in [2, 3, 4])
            
            if condicao_paridade:
                if jogo_e_valido_estritamente(palpite_ia) and palpite_ia not in jogos_finais:
                    jogos_finais.append(palpite_ia)
                    
    if len(jogos_finais) < 5:
        while len(jogos_finais) < 5:
            palpite_ia = sorted(random.sample(top_dezenas_cluster_atual, 6))
            if jogo_e_valido_estritamente(palpite_ia) and palpite_ia not in jogos_finais:
                jogos_finais.append(palpite_ia)
                    
    print("\n" + "=" * 75)
    print("      JOGOS GERADOS VIA K-MEANS CLUSTERING + REGRESSÃO LINEAR MÚLTIPLA")
    print("=" * 75)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Soma: {sum(jogo)}")
    print("=" * 75)