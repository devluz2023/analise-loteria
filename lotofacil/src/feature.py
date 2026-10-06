import pandas as pd
from collections import Counter
# Certifique-se de importar as versões adaptadas para Lotofácil
from matriz_frequencia import gerar_matriz_lotofacil 
from get_data import df # Certifique-se de baixar dados da Lotofacil
from model import colunas_bolas
from utils import LoteriaUtils

# 1. Inicialização
utils = LoteriaUtils(df)

# Geração da matriz base (agora para 25 dezenas)
df_matriz = gerar_matriz_lotofacil(df)
df_matriz.to_excel("matriz_ausencia_lotofacil.xlsx", index=False)

# Prepara mapas estatísticos
mapas_estatisticos = {
    col: dict(zip(df_matriz['Dezena'], pd.to_numeric(df_matriz[col], errors='coerce').fillna(0)))
    for col in df_matriz.columns if col != 'Dezena'
}

# 2. Processamento e Criação de Features
df['Indice'] = df.apply(lambda row: utils.calcular_indice([int(row[c]) for c in colunas_bolas]), axis=1)
ultimo_indice = df['Indice'].iloc[-1]

# Mapeamento de Elite
historico = [tuple(sorted([int(row[col]) for col in colunas_bolas])) for _, row in df.iterrows()]
freq_map = Counter(historico)
df_elite = pd.DataFrame([{'Indice': utils.calcular_indice(c), 'Combinacao': c, 'Frequencia': f} for c, f in freq_map.items()])

# Métricas com o Objeto Utils
df_elite['Distancia'] = df_elite['Indice'].apply(lambda x: utils.calcular_distancia_circular(x, ultimo_indice))
df_elite['quantidade_sequencia'] = df_elite['Combinacao'].apply(utils.maior_sequencia_circular)

# Integração da Nova Métrica: Max_Repeticao_Soma (ajuste range para 25)
mapa_max_repeticao = {d: utils.calcular_maximo_repeticao(d) for d in range(1, 26)}
df_elite['Max_Repeticao_Max'] = df_elite['Combinacao'].apply(
    lambda c: max([mapa_max_repeticao.get(n, 0) for n in c])
)

# Integração DINÂMICA de outras colunas
for nome_coluna, mapa in mapas_estatisticos.items():
    df_elite[f'{nome_coluna}_Soma'] = df_elite['Combinacao'].apply(lambda c: sum(mapa.get(n, 0) for n in c))

# 3. Filtros (Ajuste crucial para Lotofácil)
df['Similaridade_Proximo'] = [utils.calcular_acertos_proximo(i) for i in range(len(df))]
mapa_similaridade = df.groupby('Indice')['Similaridade_Proximo'].mean().to_dict()
df_elite['Similaridade_Media'] = df_elite['Indice'].map(mapa_similaridade).fillna(0)

df_resultado = df_elite[df_elite['Frequencia'] > 0].sort_values(by='Distancia')

# Filtro de Pares: Lotofácil tem 15 dezenas. 
# O equilíbrio estatístico ideal é entre 7 e 8 pares.
df_resultado = df_resultado[df_resultado['Combinacao'].apply(lambda c: 7 <= utils.contar_pares(c) <= 8)]

# 4. Exibição e Exportação
colunas_fixas = [
    'Indice', 'Combinacao', 'Frequencia', 'Distancia', 
    'quantidade_sequencia', 'Max_Repeticao_Max' , 'Soma', 'Primos', 'Moldura', 'Repetidas_Anterior', 'Similaridade_Media'
]
cols_soma_dinamicas = [c for c in df_elite.columns if c.endswith('_Soma') and c != 'Max_Repeticao_Max']
colunas_finais = colunas_fixas + cols_soma_dinamicas


# 1. CRIAR AS COLUNAS NO DATAFRAME ANTES DOS FILTROS
df_elite['Soma'] = df_elite['Combinacao'].apply(utils.calcular_soma)
df_elite['Primos'] = df_elite['Combinacao'].apply(utils.contar_primos)
df_elite['Moldura'] = df_elite['Combinacao'].apply(utils.contar_moldura)
df_elite['Repetidas_Anterior'] = df_elite['Combinacao'].apply(utils.contar_repetidas_anterior)

# 2. AGORA QUE AS COLUNAS EXISTEM, PODES FILTRAR O DF_RESULTADO
df_resultado = df_elite[df_elite['Frequencia'] > 0].copy() # Usa .copy() para evitar avisos

# Aplicar os filtros agora que as colunas existem
df_resultado = df_resultado[
    (df_resultado['Soma'].between(180, 220)) &
    (df_resultado['Primos'].between(5, 6)) &
    (df_resultado['Moldura'].between(9, 10)) &
    (df_resultado['Repetidas_Anterior'].between(8, 10))
]

# 3. ATUALIZAR A LISTA DE COLUNAS FINAIS (incluindo as novas)
colunas_novas = ['Soma', 'Primos', 'Moldura', 'Repetidas_Anterior']
colunas_finais = colunas_fixas + cols_soma_dinamicas + colunas_novas

# Agora a linha abaixo não dará erro
print(df_resultado[colunas_finais].head(15).to_string(index=False))

print("\n=== DIAGNÓSTICO LOTOFÁCIL FILTRADO ===")
print(df_resultado[colunas_finais].head(15).to_string(index=False))
df_resultado[colunas_finais].to_excel("diagnostico_lotofacil_filtrado.xlsx", index=False)