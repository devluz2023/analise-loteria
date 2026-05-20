from math import comb
import numpy as np
from neural_netowrk import model, scaler, window_size
from indice import df_mega_sena

# =============================================================
# 1. PREPARAÇÃO DOS DADOS DA ENTRADA RECENTE
# =============================================================

# Pegar os últimos 'window_size' (10) concursos do DataFrame original
ultimos_10_indices = df_mega_sena['Indice_Combinacao'].tail(window_size).values.reshape(-1, 1)

# Normalizar esses dados com o mesmo scaler usado no treino
ultimos_10_norm = scaler.transform(ultimos_10_indices)

# Preparar o formato para a Rede Neural (1 amostra, 10 janelas, 1 recurso)
entrada_predict = np.reshape(ultimos_10_norm, (1, window_size, 1))


# =============================================================
# 2. REALIZAR A PREVISÃO DO PRÓXIMO CONCURSO
# =============================================================

predicao_norm = model.predict(entrada_predict)

# Desnormalizar para obter o Índice de 1 a 50M
indice_previsto = int(scaler.inverse_transform(predicao_norm)[0][0])


# =============================================================
# 3. FUNÇÃO CORRIGIDA PARA CONVERTER ÍNDICE EM 6 NÚMEROS
# =============================================================

def indice_para_combinacao(index, n=60, k=6):
    """
    Converte um índice (base 1) de volta para os 6 números da Mega-Sena.
    Garante que o índice esteja dentro do limite antes de começar.
    """
    total_combinacoes = comb(n, k)
    
    # Proteção de limites contra pequenas variações numéricas da rede
    if index < 1:
        index = 1
    elif index > total_combinacoes:
        index = total_combinacoes

    combinacao = []
    current = 1
    
    for i in range(k):
        elementos_restantes = k - 1 - i
        
        while current <= n:
            c = comb(n - current, elementos_restantes)
            
            if index <= c:
                combinacao.append(current)
                current += 1  # O próximo número precisa ser maior que o atual
                break
            
            index -= c
            current += 1
            
    return combinacao


# Converter o índice previsto em jogo real
jogo_previsto = indice_para_combinacao(indice_previsto)

print("\n" + "-" * 50)
print(f"PREVISÃO DA REDE LSTM PARA O PRÓXIMO CONCURSO ({df_mega_sena['Concurso'].max() + 1})")
print(f"Índice Previsto pela Rede: {indice_previsto:,}")
print(f"Números Sugeridos: {jogo_previsto}")
print("-" * 50)


# =============================================================
# 4. CÁLCULO DA ACURÁCIA HISTÓRICA DO MODELO
# =============================================================

print("\nCalculando a acurácia histórica do modelo nos dados existentes...")

# Preparar todos os dados históricos para a validação
dados_totais = df_mega_sena['Indice_Combinacao'].values.reshape(-1, 1)
dados_totais_norm = scaler.transform(dados_totais)

# Criar as janelas estruturadas de 10 em 10 para todo o histórico
X_historico = []
for i in range(len(dados_totais_norm) - window_size):
    X_historico.append(dados_totais_norm[i:(i + window_size), 0])

X_historico = np.array(X_historico)
X_historico = np.reshape(X_historico, (X_historico.shape[0], X_historico.shape[1], 1))

# Rodar a previsão da rede para o histórico inteiro (sem poluir o terminal)
previsoes_historicas_norm = model.predict(X_historico, verbose=0)

# Desnormalizar os resultados da rede e coletar os gabaritos reais do histórico
previsoes_historicas = scaler.inverse_transform(previsoes_historicas_norm)
valores_reais_historicos = df_mega_sena['Indice_Combinacao'].values[window_size:].reshape(-1, 1)

# Calcular o erro absoluto de cada concurso passado
erros_absolutos = np.abs(valores_reais_historicos - previsoes_historicas)

# Definir a margem de tolerância (Sem k= para evitar TypeError no Python 3.13)
total_combinacoes = comb(60, 6)               # 50.063.860 combinações possíveis
margem_tolerancia = total_combinacoes * 0.05  # Tolerância de 5% (~2.5 milhões de posições)

# Calcular a porcentagem de acertos que ficaram dentro do raio aceitável
acertos = np.sum(erros_absolutos <= margem_tolerancia)
acuracia_historica = (acertos / len(valores_reais_historicos)) * 100

print("-" * 50)
print(f"MÉTRICA DE DESEMPENHO (Histórico Geral)")
print(f"Margem de erro aceitável: {margem_tolerancia:,.0f} posições.")
print(f"Acurácia Histórica do Modelo: {acuracia_historica:.2f}%")
print("-" * 50)