import pandas as pd
import numpy as np
from collections import Counter

def gerar_matriz_quina(df_quina):
    # 1. Configurações Iniciais Quina
    colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
    total_concursos = len(df_quina)
    total_dezenas = 80

    print(f"Iniciando mapeamento estatístico da Quina sobre {total_concursos} concursos...")

    all_absence_streaks = {d: [] for d in range(1, total_dezenas + 1)}
    atraso_atual = {d: 0 for d in range(1, total_dezenas + 1)}
    max_consecutivo = {d: 0 for d in range(1, total_dezenas + 1)}
    saiu_anterior = {d: False for d in range(1, total_dezenas + 1)}
    streak_atual = {d: 0 for d in range(1, total_dezenas + 1)}

    df_cron = df_quina.sort_values(by='Concurso').copy()

    # 2. Varrer o histórico
    for _, row in df_cron.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        
        for d in range(1, total_dezenas + 1):
            if d in sorteados:
                if atraso_atual[d] > 0:
                    all_absence_streaks[d].append(atraso_atual[d])
                    atraso_atual[d] = 0
                
                streak_atual[d] = streak_atual[d] + 1 if saiu_anterior[d] else 1
                max_consecutivo[d] = max(max_consecutivo[d], streak_atual[d])
                saiu_anterior[d] = True
            else:
                atraso_atual[d] += 1
                streak_atual[d] = 0
                saiu_anterior[d] = False

    max_observed_streak = max([max(s) if s else 0 for s in all_absence_streaks.values()] + [0])

    # 3. Construção das Colunas Estatísticas
    rows_data = []
    prob_teorica_sair = 5.0 / total_dezenas 

    for d in range(1, total_dezenas + 1):
        streaks = all_absence_streaks[d]
        counts = Counter(streaks)
        
        vezes_nao_sorteada = total_concursos - df_quina[colunas_bolas].isin([d]).any(axis=1).sum()
        prob_nao_sair_1 = vezes_nao_sorteada / total_concursos
        prob_nao_sair_max = prob_nao_sair_1 ** max_observed_streak
        
        freq_esperada = total_concursos * prob_teorica_sair
        desvio = (total_concursos - vezes_nao_sorteada) - freq_esperada

        row = {
            'Dezena': d,
            'FrequenciaMedia_Esperada': round(freq_esperada, 1),
            'Desvio': round(desvio, 1),
            'Prob_Nao_Sair_Empirica_1_Concurso': round(prob_nao_sair_1, 4),
            f'Prob_Nao_Sair_Empirica_{max_observed_streak}_Concursos': f"{prob_nao_sair_max:.6e}",
            'Máximo Consecutivo': max_consecutivo[d],
            'Total de Sequências (qualquer tamanho)': len(streaks),
            'Total de Sequências (> 1)': sum(1 for s in streaks if s > 1),
            'Máximo de Sorteios Sem Sair': max(streaks) if streaks else 0,
            'Ausencia_Actual_Concursos': atraso_atual[d]
        }
        
        for t in range(1, max_observed_streak + 1):
            row[str(t)] = counts.get(t, 0)
            
        row.update({
            'Total_Streaks_Observed': len(streaks),
            'Total_Absence_Draws': sum(streaks),
            'Average_Streak_Length': round(np.mean(streaks), 4) if streaks else 0
        })
        rows_data.append(row)

    # 4. Formatação Final
    cols_dinamicas = [str(i) for i in range(1, max_observed_streak + 1)]
    df_final = pd.DataFrame(rows_data)
    
    # Ordenar colunas para manter o padrão que você já conhece
    cols_ordem = ['Dezena', 'FrequenciaMedia_Esperada', 'Desvio', 'Prob_Nao_Sair_Empirica_1_Concurso', 
                  f'Prob_Nao_Sair_Empirica_{max_observed_streak}_Concursos', 'Máximo Consecutivo', 
                  'Total de Sequências (qualquer tamanho)', 'Total de Sequências (> 1)', 
                  'Máximo de Sorteios Sem Sair', 'Ausencia_Actual_Concursos'] + cols_dinamicas + \
                  ['Total_Streaks_Observed', 'Total_Absence_Draws', 'Average_Streak_Length']
    
    return df_final[cols_ordem]

# Para executar:
# df_matriz = gerar_matriz_quina(df_quina)
# df_matriz.to_excel("matriz_ausencia_quina.xlsx", index=False)

