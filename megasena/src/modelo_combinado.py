import pandas as pd
import numpy as np
import math
import optuna
import pulp
from sklearn.linear_model import LinearRegression, BayesianRidge
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from analise import AnaliseEstatisticaMega
# === 1. PREPARAÇÃO E SINCRONIZAÇÃO ===
df_resultado = pd.read_excel("diagnostico_megasena_filtrado.xlsx")
df_limpo = df_resultado.drop(columns=['Combinacao', 'Indice'], errors='ignore')
target = df_resultado['Indice']


class RLStrategyManager:
    def __init__(self, strategies):
        self.strategies = strategies # ['Ensemble', 'PO']
        self.q_values = {s: 0.0 for s in strategies} # Confiança inicial
        self.alpha = 0.1 # Taxa de aprendizagem
        
    def choose_strategy(self):
        # Epsilon-greedy: 90% escolhe o melhor, 10% explora
        if np.random.rand() < 0.1:
            return np.random.choice(self.strategies)
        return max(self.q_values, key=self.q_values.get)
    
    def update(self, strategy, acertos_dict):
        # Transforma o dicionário num valor único de recompensa
        # Podes ajustar os pesos conforme preferires
        reward = (acertos_dict['Terno'] * 1) + \
                 (acertos_dict['Quadra'] * 5) + \
                 (acertos_dict['Quina'] * 20) + \
                 (acertos_dict['Sena'] * 100)
        
        # Agora a conta é (float - float)
        self.q_values[strategy] += self.alpha * (reward - self.q_values[strategy])
class EnsembleRegressor:
    def __init__(self, cb_model, lstm_model, meta_model):
        self.cb = cb_model
        self.lstm = lstm_model
        self.meta = meta_model
        
    def predict(self, X_test, X_seq_test):
        # Gera previsões dos modelos base
        p_cb = self.cb.predict(X_test).flatten()
        p_lstm = self.lstm.predict(X_seq_test, verbose=0).flatten()
        # Combina as previsões no meta-modelo
        return self.meta.predict(np.column_stack((p_cb, p_lstm)))





def criar_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    # Encoder: Comprime a informação
    encoded = Dense(8, activation='relu')(input_layer)
    # Bottleneck: Onde a "normalidade" é aprendida
    bottleneck = Dense(4, activation='relu')(encoded)
    # Decoder: Tenta reconstruir o original
    decoded = Dense(8, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    ae = Model(inputs=input_layer, outputs=output_layer)
    ae.compile(optimizer='adam', loss='mse')
    return ae

# === 2. PIPELINE DE TREINO DO ENSEMBLE ===
def treinar_ensemble(X_train, X_seq_train, y_train, y_seq_train, best_params):
    # Modelos Base
    cb = CatBoostRegressor(iterations=200, verbose=False).fit(X_train, y_train)
    lstm = Sequential([Input(shape=(5, X_seq_train.shape[2])), LSTM(best_params['lstm_units']), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_seq_train, y_seq_train, epochs=20, verbose=0)
    
    # Meta-modelo (Juiz)
    p_cb = cb.predict(X_train).flatten()
    p_lstm = lstm.predict(X_seq_train, verbose=0).flatten()
    meta = LinearRegression().fit(np.column_stack((p_cb, p_lstm)), y_seq_train)
    
    return EnsembleRegressor(cb, lstm, meta)


def create_sequences(data, target, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i : i + seq_length].values)
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df_limpo, target)
indices = np.arange(len(X_seq))
train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=False)

X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
y_seq_train, y_seq_test = y_seq[train_idx], y_seq[test_idx]
X_train, X_test = df_limpo.iloc[train_idx + 5], df_limpo.iloc[test_idx + 5]
y_train, y_test = target.iloc[train_idx + 5], target.iloc[test_idx + 5]



# --- Adicione esta função auxiliar para converter o índice ---
def indice_para_combinacao(n, k, index):
    comb = []; start = 1
    for i in range(k):
        for val in range(start, n + 1):
            count = math.comb(n - val, k - 1 - i)
            if index < count: comb.append(val); start = val + 1; break
            else: index -= count
    return comb


def eh_jogo_normal(jogo_dezenas, threshold=0.05):
    # Converte o jogo em formato de features (mesmo formato que o df_limpo)
    # Aqui assumimos uma transformação simplificada para exemplo
    features_jogo = np.zeros((1, dados_norm.shape[1])) 
    # ... (preencha as colunas do seu df_limpo com os dados do jogo) ...
    
    jogo_norm = scaler_ae.transform(features_jogo)
    reconstrucao = autoencoder.predict(jogo_norm, verbose=0)
    erro = np.mean(np.power(jogo_norm - reconstrucao, 2))
    
    return erro < threshold # Retorna True se o jogo for "normal"


# === 2. FUNÇÕES DE APOIO ===
def avaliar(preds, y_target, nome):
    """
    Avalia a performance de modelos de loteria contando acertos reais.
    preds: Array de índices previstos pelo modelo.
    y_target: Array de índices reais do sorteio.
    """
    preds = preds.flatten()
    total = len(y_target)
    acertos = {"Terno": 0, "Quadra": 0, "Quina": 0, "Sena": 0}
    
    for pred, actual in zip(preds, y_target):
        # Converte índice para dezenas (o formato real do jogo)
        p = set(indice_para_combinacao(60, 6, int(max(0, min(float(pred), 50063859)))))
        r = set(indice_para_combinacao(60, 6, int(actual)))
        
        # Conta interseção
        inter = len(p.intersection(r))
        if inter >= 3:
            nome_acerto = ["", "", "", "Terno", "Quadra", "Quina", "Sena"][inter]
            acertos[nome_acerto] += 1
            
    # Relatório visual
    mse = mean_squared_error(y_target, preds)
    print(f"\n{'='*15} PERFORMANCE: {nome} {'='*15}")
    print(f"MSE (Erro): {mse:.2f}")
    for k, v in acertos.items():
        print(f"Acertos de {k}: {v} ({ (v/total)*100:.2f}% dos jogos testados)")
    print(f"{'='*50}")

def modelo_po_probabilistico(df_historico):
    """
    Modelo de PO que seleciona dezenas baseadas na probabilidade histórica
    com restrições de soma e paridade.
    """
    # 1. Limpeza e conversão dos dados (Corrigindo erro de str/int)
    # Garante que apenas as colunas de dezenas sejam lidas como numéricas
    df_dezenas = df_historico.iloc[:, :6] 
    dezenas_flat = df_dezenas.apply(pd.to_numeric, errors='coerce').values.flatten()
    dezenas_flat = dezenas_flat[~np.isnan(dezenas_flat)].astype(int)
    
    # Calcular probabilidades (frequência histórica)
    frequencias = pd.Series(dezenas_flat).value_counts(normalize=True).sort_index()
    
    # Mapear frequências para os 60 números possíveis
    pesos = [frequencias.get(i, 0.0001) for i in range(1, 61)]
    
    # 2. Definição do Problema (Maximização da probabilidade total)
    prob = pulp.LpProblem("Selecao_Probabilistica_PO", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("d", range(1, 61), cat=pulp.LpBinary)
    
    # Função Objetivo: Maximizar a soma dos pesos das dezenas escolhidas
    prob += pulp.lpSum([pesos[i-1] * x[i] for i in range(1, 61)])
    
    # 3. Restrições do Sistema
    # Exatamente 6 números
    prob += pulp.lpSum([x[i] for i in range(1, 61)]) == 6
    
    # Soma total entre 160 e 240 (padrão estatístico da Mega)
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) >= 160
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) <= 240
    
    # Equilíbrio entre pares e ímpares (pelo menos 2 de cada)
    prob += pulp.lpSum([x[i] for i in range(1, 61) if i % 2 == 0]) >= 2
    prob += pulp.lpSum([x[i] for i in range(1, 61) if i % 2 != 0]) >= 2
    
    # 4. Solução
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    return sorted([i for i in range(1, 61) if pulp.value(x[i]) == 1])

# === EXECUÇÃO ===
# df_resultado deve ser o teu DataFrame carregado com os dados históricos
try:
    sugestao_po = modelo_po_probabilistico(df_resultado)
    print(f"Sugestão Probabilística (PO): {sugestao_po}")
except Exception as e:
    print(f"Erro ao executar o modelo PO: {e}")

def otimizar_com_pulp(indice_alvo):
    media_alvo = (indice_alvo / 50063859) * 60
    prob = pulp.LpProblem("Otimizacao_Final", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("d", range(1, 61), cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in range(1, 61)]) == 6
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) >= 160
    prob += pulp.lpSum([i * x[i] for i in range(1, 61)]) <= 240
    prob += pulp.lpSum([x[i] * abs(i - media_alvo) for i in range(1, 61)])
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sorted([i for i in range(1, 61) if pulp.value(x[i]) == 1])

# === 3. OTIMIZAÇÃO HÍBRIDA ===
# Grid Search para Bayes
grid_bayes = GridSearchCV(BayesianRidge(), {'alpha_1': [1e-6, 1e-4], 'lambda_1': [1e-6, 1e-4]}, cv=3)
grid_bayes.fit(X_train, y_train)


def calcular_acertos_po(historico_real, sugestoes_po):
    # Converte os índices reais em conjuntos de dezenas para comparar
    total_testes = len(historico_real)
    acertos = {"Terno": 0, "Quadra": 0, "Quina": 0, "Sena": 0}
    
    # Historico_real aqui deve ser o target (dezenas) do teste
    for dezenas_reais in historico_real:
        # sugestao_po é a lista fixa gerada pelo solver
        inter = len(set(sugestao_po).intersection(set(dezenas_reais)))
        if inter >= 3:
            nome_acerto = ["", "", "", "Terno", "Quadra", "Quina", "Sena"][inter]
            acertos[nome_acerto] += 1
            
    print(f"\n--- Performance: PO Probabilístico ---")
    for k, v in acertos.items():
        print(f"{k}: {v} acertos ({v/total_testes*100:.2f}%)")

# Optuna para Ensemble
def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 32, 64)
    cb = CatBoostRegressor(iterations=200, verbose=False).fit(X_train, y_train)
    lstm = Sequential([Input(shape=(5, X_seq.shape[2])), LSTM(lstm_units), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_seq_train, y_seq_train, epochs=10, verbose=0)
    meta = LinearRegression().fit(np.column_stack((cb.predict(X_train), lstm.predict(X_seq_train, verbose=0).flatten())), y_seq_train)
    preds = meta.predict(np.column_stack((cb.predict(X_test), lstm.predict(X_seq_test, verbose=0).flatten())))
    return mean_squared_error(y_seq_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# === 4. COMPARAÇÃO E RESULTADOS ===
model_lr = LinearRegression().fit(X_train, y_train)
model_bayes = grid_bayes.best_estimator_
model_cb = CatBoostRegressor(iterations=200, verbose=False).fit(X_train, y_train)
model_lstm = Sequential([Input(shape=(5, X_seq.shape[2])), LSTM(study.best_params['lstm_units']), Dense(1)])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_seq_train, y_seq_train, epochs=20, verbose=0)
model_ensemble = treinar_ensemble(X_train, X_seq_train, y_train, y_seq_train, study.best_params)

# Agora podes comparar o Ensemble com os outros no teu Avaliador:
avaliar(model_ensemble.predict(X_test, X_seq_test), y_seq_test, "Ensemble Stacking")
# Avaliar
avaliar(model_lr.predict(X_test), y_test, "Linear")
avaliar(model_bayes.predict(X_test), y_test, "Bayesian Grid")
avaliar(model_cb.predict(X_test), y_test, "CatBoost")

# === 5. SUGESTÃO FINAL (ENSEMBLE) ===
meta = LinearRegression().fit(np.column_stack((model_cb.predict(X_train), model_lstm.predict(X_seq_train, verbose=0).flatten())), y_seq_train)
p_cb_f = model_cb.predict(df_limpo.iloc[[-1]]).flatten()
p_lstm_f = model_lstm.predict(np.array([df_limpo.iloc[-5:].values]), verbose=0).flatten()
pred_final = meta.predict(np.column_stack((p_cb_f, p_lstm_f)))
y_test_dezenas = df_resultado.iloc[test_idx + 5].iloc[:, :6].values.tolist()

# Executar a avaliação
sugestao_po = modelo_po_probabilistico(df_resultado) # O modelo que já criamos
print(f"\nSugestão PO: {sugestao_po}")
calcular_acertos_po(y_test_dezenas, sugestao_po)
print(f"\nSugestão Final (Ensemble + PuLP): {otimizar_com_pulp(pred_final[0])}")


manager = RLStrategyManager(['Ensemble', 'PO'])

def avaliar_trio(sugestao, sorteio_real):
    inter = len(set(sugestao).intersection(set(sorteio_real)))
    res = {"Terno": 0, "Quadra": 0, "Quina": 0, "Sena": 0}
    if inter >= 3:
        nome = ["", "", "", "Terno", "Quadra", "Quina", "Sena"][inter]
        res[nome] = 1
    return res

# Simulando o Teste A/B com aprendizado RL
print("\n--- Iniciando Treino do Agente RL ---")
for i in range(len(y_test_dezenas)):
    strategy = manager.choose_strategy()
    
    # Obter previsão
    if strategy == 'Ensemble':
        # Converte índice previsto para dezenas
        idx = int(max(0, min(float(model_ensemble.predict(X_test, X_seq_test)[i]), 50063859)))
        sugestao = indice_para_combinacao(60, 6, idx)
    else:
        sugestao = sugestao_po
        
    # Atualiza o Agente com o resultado real
    reward = avaliar_trio(sugestao, y_test_dezenas[i])
    manager.update(strategy, reward)

print(f"Confiança final do Agente RL: {manager.q_values}")
vencedor = max(manager.q_values, key=manager.q_values.get)
print(f"Estratégia Recomendada pelo RL: {vencedor}")

# Sugestão Final baseada no vencedor
if vencedor == 'Ensemble':
    print(f"Sugestão Final: {otimizar_com_pulp(pred_final[0])}")
else:
    print(f"Sugestão Final: {sugestao_po}")



def extrair_features(dezenas, df_modelo):
    """
    Transforma as 6 dezenas nas mesmas colunas estatísticas do teu df_limpo.
    """
    # Exemplo: Se o teu df_limpo tem Soma, Paridade, etc...
    # Tens de extrair exatamente na mesma ordem!
    features = [
        sum(dezenas),                          # Soma
        len([d for d in dezenas if d % 2 == 0]),# Pares
        # ... adiciona aqui todas as outras colunas que tens no df_limpo ...
    ]
    # Retorna como um array para o scaler
    return np.array(features).reshape(1, -1)




# Supondo que 'sugestao_ml' seja a lista de 6 dezenas que o Ensemble gerou
def filtrar_jogo(sugestao_ml, racional_df):
    jogo_final = []
    
    # Garante que o racional_df tenha o índice como inteiro
    racional_df.index = racional_df.index.astype(int)
    
    for dezena in sugestao_ml:
        try:
            # Acede ao valor garantindo que a dezena é um inteiro
            racional_valor = racional_df.loc[int(dezena), 'Racional']
            
            if racional_valor == 'Tendencia_Baixa':
                # Busca uma substituta que seja 'Tendencia_Alta'
                # Garantimos que a substituta seja também um inteiro
                substitutas = racional_df[racional_df['Racional'] == 'Tendencia_Alta'].index
                if len(substitutas) > 0:
                    substituta = np.random.choice(substitutas)
                    jogo_final.append(int(substituta))
                else:
                    jogo_final.append(int(dezena)) # Se não achar, mantém a original
            else:
                jogo_final.append(int(dezena))
                
        except KeyError:
            # Se a dezena não estiver no racional (ex: dezena nova), mantém
            jogo_final.append(int(dezena))
            
    return sorted(list(set(jogo_final)))


analise = AnaliseEstatisticaMega(df_resultado)
racional_estatistico = analise.encontrar_racional()
print(racional_estatistico.head(10))


jogo_filtrado = filtrar_jogo(otimizar_com_pulp(pred_final[0]), racional_estatistico)
print(f"=== O JOGO RECOMENDADO PARA HOJE ===")
print(f"Dezenas: {jogo_filtrado}")

# No final do seu código:
sugestao_bruta = otimizar_com_pulp(pred_final[0])
jogo_filtrado_analise = filtrar_jogo(sugestao_bruta, racional_estatistico)
print(f"\nSugestão enconders: {jogo_filtrado_analise}")



scaler = MinMaxScaler()
dados_norm = scaler.fit_transform(df_limpo.values)
autoencoder = criar_autoencoder(dados_norm.shape[1])
autoencoder.fit(dados_norm, dados_norm, epochs=100, verbose=0)

# 3. Função de Validação do Jogo
def validar_jogo_autoencoder(jogo_dezenas, df_limpo, threshold=0.05):
    # 1. CRIAR O VETOR VAZIO: Um array com 1 linha e 124 colunas (tamanho do seu df)
    features = np.zeros((1, df_limpo.shape[1]))
    
    # 2. PREENCHER AS COLUNAS (Exemplo de mapeamento)
    # Aqui tu tens de garantir que as colunas que calculas 
    # batem exatamente com as colunas do teu df_limpo.
    
    # Exemplo: se a coluna 0 for 'Soma'
    features[0, 0] = sum(jogo_dezenas)
    # Exemplo: se a coluna 1 for 'Pares'
    features[0, 1] = len([d for d in jogo_dezenas if d % 2 == 0])
    
    # Se tiveres outras colunas, tens de continuar a preencher:
    # features[0, 2] = ... (alguma outra estatística)
    
    # 3. AGORA O SCALER FUNCIONA: Ele recebe 124 colunas como esperado
    features_norm = scaler.transform(features)
    
    # 4. Reconstrói e calcula o erro
    reconstrucao = autoencoder.predict(features_norm, verbose=0)
    erro = np.mean(np.power(features_norm - reconstrucao, 2))
    
    return erro < threshold, erro

# 4. Uso final
# O 'jogo_filtrado' é o teu jogo após passar pelos outros filtros
 # Passa o DataFrame inteiro (df_limpo), sem o .columns
is_valid, erro_valor = validar_jogo_autoencoder(jogo_filtrado, df_limpo)

if is_valid:
    print(f"✅ Jogo Aceite (Erro: {erro_valor:.4f})")
else:
    print(f"⚠️ Jogo Rejeitado (Erro: {erro_valor:.4f} > Threshold)")

    # --- Integração no teu pipeline ---
