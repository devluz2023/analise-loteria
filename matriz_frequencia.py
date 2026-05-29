import pandas as pd
import numpy as np
from collections import Counter
from get_data import df_mega_sena

def gerar_matriz():
        
    # 1. Configurações Iniciais
    colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
    total_concursos = len(df_mega_sena)

    print(f"Iniciando mapeamento estatístico sobre {total_concursos} concursos...")

    # Dicionários para armazenar os históricos de jejuns
    all_absence_streaks_per_dezena = {dezena: [] for dezena in range(1, 61)}
    atraso_atual = {dezena: 0 for dezena in range(1, 61)}
    max_consecutivo_sorteios = {dezena: 0 for dezena in range(1, 61)} # Vezes que saiu em concursos seguidos

    # Auxiliar para contar saídas consecutivas
    saiu_no_anterior = {dezena: False for dezena in range(1, 61)}
    streak_consecutivo_atual = {dezena: 0 for dezena in range(1, 61)}

    # Ordenar os concursos do primeiro ao último
    df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

    # 2. Varrer o histórico para coletar os tamanhos exatos de todos os jejuns
    for index, row in df_cronologico.iterrows():
        sorteados = set([int(row[c]) for c in colunas_bolas])
        
        for dezena in range(1, 61):
            if dezena in sorteados:
                # Se estava sem sair, fecha o ciclo de ausência e guarda o tamanho dele
                if atraso_atual[dezena] > 0:
                    all_absence_streaks_per_dezena[dezena].append(atraso_atual[dezena])
                    atraso_atual[dezena] = 0
                
                # Controle de saídas consecutivas (saiu jogo seguido)
                if saiu_no_anterior[dezena]:
                    streak_consecutivo_atual[dezena] += 1
                else:
                    streak_consecutivo_atual[dezena] = 1
                
                if streak_consecutivo_atual[dezena] > max_consecutivo_sorteios[dezena]:
                    max_consecutivo_sorteios[dezena] = streak_consecutivo_atual[dezena]
                    
                saiu_no_anterior[dezena] = True
            else:
                # Se não saiu, acumula o jejum atual
                atraso_atual[dezena] += 1
                streak_consecutivo_atual[dezena] = 0
                saiu_no_anterior[dezena] = False

    # Descobrir qual foi o maior jejum que aconteceu na história com qualquer dezena
    max_observed_streak = 0
    for dezena, streaks in all_absence_streaks_per_dezena.items():
        if streaks:
            max_observed_streak = max(max_observed_streak, max(streaks))

    # 3. Construção das Colunas Estatísticas e de Resumo
    rows_data = []

    # Probabilidade teórica de uma dezena NÃO sair em 1 concurso: 54/60 = 0.90
    prob_teorica_nao_sair = 54.0 / 60.0 

    for dezena in range(1, 61):
        streaks = all_absence_streaks_per_dezena[dezena]
        streak_counts = Counter(streaks)
        
        # Contagens de sequências
        total_sequences_any = len(streaks)
        total_sequences_greater_1 = sum(1 for s in streaks if s > 1)
        
        total_absence_draws = sum(streaks)
        average_streak_len = total_absence_draws / total_sequences_any if total_sequences_any > 0 else 0
        
        # Cálculos de Probabilidade Empírica (Baseada no Histórico Real)
        # Total de vezes que a dezena NÃO saiu na história inteira dividido pelo total de concursos
        vezes_nao_sorteada = total_concursos - df_mega_sena[colunas_bolas].isin([dezena]).any(axis=1).sum()
        prob_nao_sair_1_concurso = vezes_nao_sorteada / total_concursos
        prob_nao_sair_max_concursos = prob_nao_sair_1_concurso ** max_observed_streak
        
        # Frequência Média Esperada e Desvio Padrão Teórico
        frequencia_media_esperada = total_concursos * (6.0 / 60.0) # Esperado de saídas = 10% dos concursos
        saidas_reais = total_concursos - vezes_nao_sorteada
        desvio_padrao_observado = saidas_reais - frequencia_media_esperada

        # Monta a linha inicial com os cabeçalhos estatísticos idênticos ao seu modelo
        row_entry = {
            'Dezena': dezena,
            'FrequenciaMedia_Esperada': round(frequencia_media_esperada, 1),
            'Desvio': round(desvio_padrao_observado, 1),
            'Prob_Nao_Sair_Empirica_1_Concurso': round(prob_nao_sair_1_concurso, 4),
            f'Prob_Nao_Sair_Empirica_{max_observed_streak}_Concursos': f"{prob_nao_sair_max_concursos:.6e}",
            'Máximo Consecutivo': max_consecutivo_sorteios[dezena],
            'Total de Sequências (qualquer tamanho)': total_sequences_any,
            'Total de Sequências (> 1)': total_sequences_greater_1,
            'Máximo de Sorteios Sem Sair': max(streaks) if streaks else 0,
            'Ausencia_Actual_Concursos': atraso_atual[dezena]
        }
        
        # Preenche a contagem de vezes para cada tamanho de jejum (Coluna 1 até o Máximo Histórico)
        for streak_len in range(1, max_observed_streak + 1):
            row_entry[str(streak_len)] = streak_counts.get(streak_len, 0)
            
        # Colunas finais de consolidação
        row_entry['Total_Streaks_Observed'] = total_sequences_any
        row_entry['Total_Absence_Draws'] = total_absence_draws
        row_entry['Average_Streak_Length'] = round(average_streak_len, 4)
        
        rows_data.append(row_entry)

    # 4. Organizar Ordem das Colunas na Tabela Final
    colunas_estatisticas = [
        'Dezena', 'FrequenciaMedia_Esperada', 'Desvio', 
        'Prob_Nao_Sair_Empirica_1_Concurso', f'Prob_Nao_Sair_Empirica_{max_observed_streak}_Concursos',
        'Máximo Consecutivo', 'Total de Sequências (qualquer tamanho)', 'Total de Sequências (> 1)', 
        'Máximo de Sorteios Sem Sair', 'Ausencia_Actual_Concursos'
    ]
    colunas_dinamicas_jejum = [str(i) for i in range(1, max_observed_streak + 1)]
    colunas_resumo_finais = ['Total_Streaks_Observed', 'Total_Absence_Draws', 'Average_Streak_Length']

    colunas_finais_ordenadas = colunas_estatisticas + colunas_dinamicas_jejum + colunas_resumo_finais

    # Criar DataFrame estruturado
    df_matriz_completa = pd.DataFrame(rows_data, columns=colunas_finais_ordenadas)

    # 5. Salvar em Excel para você analisar perfeitamente sem quebras de linha
    nome_arquivo = "matriz_ausencia_completa.xlsx"
    df_matriz_completa.to_excel(nome_arquivo, index=False)

 