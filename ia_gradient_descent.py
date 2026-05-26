import numpy as np
import pandas as pd
import random
import optuna
import os
from math import log
from collections import Counter
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from gerador import jogo_e_valido_estritamente
from indice import encontrar_posicao_itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 80)
print("  INICIALIZANDO MOTOR DE OTIMIZAÇÃO VIA GRADIENTE DESCENDENTE (GD)")
print("=" * 80)

colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()
total_sorteios = len(df_cronologico)

# Divisão de Dados para Prova de Acurácia (80% Histórico Base, 20% Bloco Oculto de Teste)
split_idx = int(total_sorteios * 0.8)
df_treino_base = df_cronologico.iloc[:split_idx].copy()
df_teste_validacao = df_cronologico.iloc[split_idx:].copy()


# =============================================================
# 1. MOTOR DE PROVIMENTO HISTÓRICO E METRICAS DE LOSS
# =============================================================
def extrair_ambiente_geometrico_historico(df_historico_ponto):
    """ Calcula o centro de gravidade e as forças de atração a priori """
    sorteios_locais = len(df_historico_ponto)
    matriz_v_6d = df_historico_ponto[colunas_bolas].values.astype(np.float32)
    vetor_mu_local = np.mean(matriz_v_6d, axis=0)
    
    dist_euclidianas = np.sqrt(np.sum((matriz_v_6d - vetor_mu_local) ** 2, axis=1))
    media_d, desvio_d = np.mean(dist_euclidianas), np.std(dist_euclidianas)
    lim_min = media_d - (0.6 * desvio_d)
    lim_max = media_d + (1.2 * desvio_d)
    
    current_ausencia = {d: 0 for d in range(1, 61)}
    todas_dezenas_locais = []
    for index, row in df_historico_ponto.iterrows():
        sorteados = set([row[col] for col in colunas_bolas])
        for d in range(1, 61):
            if d in sorteados: current_ausencia[d] = 0
            else: current_ausencia[d] += 1
            if d in sorteados: todas_dezenas_locais.append(d)
            
    frequencia_dezenas = Counter(todas_dezenas_locais)
    
    score_dezenas_local = {}
    for d in range(1, 61):
        freq = frequencia_dezenas.get(d, 0)
        desvio = freq - (sorteios_locais / 10)
        jejum = current_ausencia[d]
        p_bayes = max(0.0001, (freq / (sorteios_locais + 1e-5)) * (1 / (1 + jejum)))
        score_dezenas_local[d] = (0.5 * log(p_bayes)) + (0.3 * (freq / (sorteios_locais + 1e-5))) + (0.2 * (desvio ** 2) / (sorteios_locais + 1e-5))
        
    return vetor_mu_local, lim_min, lim_max, score_dezenas_local


def calcular_funcao_custo(vetor_continuo, score_dezenas, v_mu, l_min, l_max):
    """
    Função de Custo (Loss Function): Mede o erro do vetor em relação 
    à casca geométrica e às propensões de Bayes. Queremos minimizar este valor.
    """
    loss = 0.0
    
    # 1. Erro de barreira física: força o vetor a manter-se no limite do volante [1, 60]
    for coord in vetor_continuo:
        if coord < 1.0: loss += 200.0 * (1.0 - coord) ** 2
        elif coord > 60.0: loss += 200.0 * (coord - 60.0) ** 2
        
    # 2. Erro de Ordenação Estrita (b1 < b2 < ... < b6)
    for i in range(5):
        if vetor_continuo[i] >= vetor_continuo[i+1]:
            loss += 150.0 * (vetor_continuo[i] - vetor_continuo[i+1] + 1.0) ** 2
            
    # 3. Erro da Casca do Hipercubo (Distância Euclidiana)
    dist_p = np.sqrt(np.sum((vetor_continuo - v_mu) ** 2))
    if dist_p < l_min:
        loss += 1.5 * (l_min - dist_p) ** 2
    elif dist_p > l_max:
        loss += 1.5 * (dist_p - l_max) ** 2
        
    # 4. Erro da Curva de Grandeza da Soma (Target ideal no miolo central = 160)
    soma_atual = np.sum(vetor_continuo)
    if not (120 <= soma_atual <= 200):
        loss += 0.5 * (soma_atual - 160.0) ** 2
        
    # 5. Penalização baseada nas notas de Bayes (Mapeamento contínuo aproximado)
    for coord in vetor_continuo:
        num_discreto = int(np.clip(round(coord), 1, 60))
        nota_bayes = score_dezenas.get(num_discreto, -5.0)
        loss -= 8.0 * nota_bayes # Subtrair a nota equivale a minimizar o inverso (maximizar atração)
        
    return loss


def computar_gradiente_numerico(vetor, score_dezenas, v_mu, l_min, l_max, epsilon=1e-3):
    """ Calcula as derivadas parciais (Gradiente) através de diferenças finitas """
    gradiente = np.zeros_like(vetor)
    for i in range(len(vetor)):
        vetor_positivo = vetor.copy()
        vetor_positivo[i] += epsilon
        loss_positivo = calcular_funcao_custo(vetor_positivo, score_dezenas, v_mu, l_min, l_max)
        
        vetor_negativo = vetor.copy()
        vetor_negativo[i] -= epsilon
        loss_negativo = calcular_funcao_custo(vetor_negativo, score_dezenas, v_mu, l_min, l_max)
        
        # Fórmula do gradiente centrado: (f(x+e) - f(x-e)) / 2e
        gradiente[i] = (loss_positivo - loss_negativo) / (2.0 * epsilon)
        
    return gradiente


def otimizar_por_gradiente_descendente(score_dezenas, v_mu, l_min, l_max, lr=0.02, passos=120):
    """ Executa a descida do gradiente para lapidar o vetor em direção ao custo mínimo """
    jogos_otimizados = []
    
    # Tentamos partir de 15 chutes iniciais diferentes espalhados pelo espaço para achar mínimos distintos
    for _ in range(15):
        # Chute inicial aleatório ordenado
        vetor_atual = np.array(sorted(random.sample(range(1, 61), 6)), dtype=np.float32)
        
        for _ in range(passos):
            grad = computar_gradiente_numerico(vetor_atual, score_dezenas, v_mu, l_min, l_max)
            # Clip do gradiente para evitar explosão vetorial
            grad = np.clip(grad, -15.0, 15.0)
            
            # PASSO DO GRADIENTE: atualiza na direção oposta à subida do erro
            vetor_atual -= lr * grad
            
        # Concretiza o vetor contínuo de volta para dezenas discretas inteiras
        jogo_final_discreto = []
        for coord in vetor_atual:
            num = int(np.clip(round(coord), 1, 60))
            if num not in jogo_final_discreto:
                jogo_final_discreto.append(num)
                
        # Garante o preenchimento de segurança com 6 dezenas e ordena
        while len(jogo_final_discreto) < 6:
            suplementar = random.randint(1, 60)
            if suplementar not in juego_final_discreto if 'juego_final_discreto' in locals() else jogo_final_discreto:
                jogo_final_discreto.append(suplementar)
        jogo_final_discreto = sorted(jogo_final_discreto)
        
        # Filtro de validação rígida na saída do otimizador
        if jogo_e_valido_estritamente(jogo_final_discreto) and jogo_final_discreto not in jogos_otimizados:
            if sum(1 for n in jogo_final_discreto if n % 2 == 0) in [2, 3, 4]:
                jogos_otimizados.append(jogo_final_discreto)
                if len(jogos_otimizados) == 5: break
                
    return jogos_otimizados


# =============================================================
# 2. FUNÇÃO OBJETIVO DO OPTUNA
# =============================================================
def objective(trial):
    lr = trial.suggest_float('lr', 0.005, 0.08, log=True)
    passos = trial.suggest_int('passos', 60, 150)
    
    # Validação rápida de performance para o trial
    v_mu_p, l_min_p, l_max_p, genes_p = extrair_ambiente_geometrico_historico(df_treino_base)
    jogos_teste = otimizar_por_gradiente_descendente(genes_p, v_mu_p, l_min_p, l_max_p, lr=lr, passos=passos)
    
    # O objetivo é maximizar a quantidade de combinações estruturalmente válidas encontradas pelo GD
    return len(jogos_teste)


# =============================================================
# 3. PROVA DE ACURÁCIA REAL (BACKTESTING VIA GRADIENTE)
# =============================================================
if __name__ == '__main__':
    print("\n[FASE 1] Otimizando hiperparâmetros da taxa de descida (Learning Rate)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=6)
    
    params = study.best_params
    print(f"\n-> Parâmetros de Gradiente Campeões: {params}")
    
    print(f"\n[FASE 2] Calculando Acurácia Real em {len(df_teste_validacao)} concursos com descida de gradiente...")
    total_ternos, total_quadras, total_quinas, total_senas = 0, 0, 0, 0
    total_bilhetes_testados = 0
    
    for i in range(len(df_teste_validacao)):
        concurso_alvo = df_teste_validacao.iloc[i]
        gabarito_real = set([int(concurso_alvo[col]) for col in colunas_bolas])
        
        df_historico_ponto = df_cronologico[df_cronologico['Concurso'] < concurso_alvo['Concurso']]
        
        # Foto estatística e geométrica do passado
        v_mu_ponto, l_min_ponto, l_max_ponto, dezenas_ponto = extrair_ambiente_geometrico_historico(df_historico_ponto)
        
        # Executa o algoritmo de descida simulando a época
        jogos_gerados_ponto = otimizar_por_gradiente_descendente(
            dezenas_ponto, v_mu_ponto, l_min_ponto, l_max_ponto, 
            lr=params['lr'], passos=params['passos']
        )
        
        for jogo in jogos_gerados_ponto:
            total_bilhetes_testados += 1
            acertos = len(set(jogo) & gabarito_real)
            if acertos == 6: total_senas += 1
            elif acertos == 5: total_quinas += 1
            elif acertos == 4: total_quadras += 1
            elif acertos == 3: total_ternos += 1

    taxa_ternos = (total_ternos / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0
    taxa_quadras = (total_quadras / total_bilhetes_testados) * 100 if total_bilhetes_testados > 0 else 0

    print("-" * 85)
    print(f"RELATÓRIO DE ACURÁCIA DO OTIMIZADOR DE GRADIENTE DESCENDENTE:")
    print(f"▶ Total de bilhetes extraídos por convergência de custo mínimo: {total_bilhetes_testados}")
    print("-" * 85)
    print("TAXA PERCENTUAL DE CONVERSÃO HISTÓRICA (HIT RATE DO GRADIENTE):")
    print(f"▶ TERNOS:  {total_ternos} acertos ➔ {taxa_ternos:.4f}%")
    print(f"▶ QUADRAS: {total_quadras} acertos ➔ {taxa_quadras:.4f}%")
    print(f"▶ QUINAS:  {total_quinas} acertos")
    print("-" * 85)
    
    # =============================================================
    # 4. GERAÇÃO DO PALPITE DO PRÓXIMO CONCURSO REAL
    # =============================================================
    print("\n[FASE 3] Lapidando a curva de erro para o PRÓXIMO concurso real...")
    v_mu_atual, l_min_atual, l_max_atual, dezenas_atual = extrair_ambiente_geometrico_historico(df_cronologico)
    
    # Roda a descida de custo definitiva para hoje
    jogos_finais = otimizar_por_gradiente_descendente(
        dezenas_atual, v_mu_atual, l_min_atual, l_max_atual, 
        lr=params['lr'], passos=params['passos']
    )
    
    print("\n" + "=" * 85)
    print("      JOGOS DEFINITIVOS EMITIDOS VIA CONVERGÊNCIA DE GRADIENTE DESCENDENTE")
    print("=" * 85)
    for i, jogo in enumerate(jogos_finais, 1):
        jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
        linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
        
        dist_final = np.sqrt(np.sum((np.array(jogo) - v_mu_atual) ** 2))
        indice_calculado = encontrar_posicao_itertools(linha_pandas)
        indice_br = f"{indice_calculado:,}".replace(",", ".")
        
        print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Distância 6D: {dist_final:.2f} | Soma: {sum(jogo)}")
    print("=" * 85)