import pandas as pd
from collections import Counter
from matriz_frequencia import gerar_matriz_megasena 
from get_data import df
from model import colunas_bolas
from utils import LoteriaUtils

# 1. Inicialização
utils = LoteriaUtils(df)

# Geração da matriz base
df_matriz = gerar_matriz_megasena(df)
df_matriz.to_excel("matriz_ausencia_megasena.xlsx", index=False)

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

# Integração da Nova Métrica: Max_Repeticao_Soma
mapa_max_repeticao = {d: utils.calcular_maximo_repeticao(d) for d in range(1, 61)}
df_elite['Max_Repeticao_Max'] = df_elite['Combinacao'].apply(
    lambda c: max([mapa_max_repeticao.get(n, 0) for n in c])
)

# Integração DINÂMICA de outras colunas
for nome_coluna, mapa in mapas_estatisticos.items():
    df_elite[f'{nome_coluna}_Soma'] = df_elite['Combinacao'].apply(lambda c: sum(mapa.get(n, 0) for n in c))

# 3. Filtros
df['Similaridade_Proximo'] = [utils.calcular_acertos_proximo(i) for i in range(len(df))]
mapa_similaridade = df.groupby('Indice')['Similaridade_Proximo'].mean().to_dict()
df_elite['Similaridade_Media'] = df_elite['Indice'].map(mapa_similaridade).fillna(0)

df_resultado = df_elite[df_elite['Frequencia'] > 0].sort_values(by='Distancia')
df_resultado = df_resultado[df_resultado['Combinacao'].apply(lambda c: 2 <= utils.contar_pares(c) <= 3)]

# 4. Definição Final das colunas para Exibição
colunas_fixas = [
    'Indice', 'Combinacao', 'Frequencia', 'Distancia', 
    'quantidade_sequencia', 'Max_Repeticao_Max' # <--- AQUI A COLUNA APARECE
]
cols_soma_dinamicas = [c for c in df_elite.columns if c.endswith('_Soma') and c != 'Max_Repeticao_Max']
colunas_finais = colunas_fixas + cols_soma_dinamicas

# Exibição e Exportação
print("\n=== DIAGNÓSTICO FILTRADO ===")
print(df_resultado[colunas_finais].head(15).to_string(index=False))
df_resultado[colunas_finais].to_excel("diagnostico_megasena_filtrado.xlsx", index=False)