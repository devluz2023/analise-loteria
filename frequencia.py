from itertools import combinations
import pandas as pd
from get_data import df_mega_sena

# 1. Extrair os jogos de forma limpa
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
sorteios = {}

print("Carregando e organizando os concursos...")
for index, row in df_mega_sena.iterrows():
    concurso = int(row['Concurso'])
    # Guardamos os números como um 'set' (conjunto) para fazer interseção rápida depois
    dezenas = set([int(row[c]) for c in colunas_bolas])
    sorteios[concurso] = dezenas

# Dicionários para guardar o mapeamento de quais concursos repetiram o quê
# Chave: a combinação que repetiu | Valor: lista com os números dos concursos
repeticoes = {6: {}, 5: {}, 4: {}, 3: {}, 2: {}}

print("Analisando a interseção par a par entre todos os concursos... Isso pode levar alguns segundos.")

lista_concursos = sorted(list(sorteios.keys()))
total_concursos = len(lista_concursos)

# 2. Comparar cada concurso com todos os seguintes (evita duplicar A vs B e B vs A)
for i in range(total_concursos):
    c1 = lista_concursos[i]
    jogo1 = sorteios[c1]
    
    for j in range(i + 1, total_concursos):
        c2 = lista_concursos[j]
        jogo2 = sorteios[c2]
        
        # Encontra os números que aparecem nos dois jogos ao mesmo tempo
        numeros_em_comum = jogo1 & jogo2
        quantidade_comum = len(numeros_em_comum)
        
        # Se compartilham de 2 a 6 números, guardamos essa ocorrência de forma exclusiva
        if 2 <= quantidade_comum <= 6:
            # Transformamos em tupla ordenada para servir de chave única no dicionário
            comb_comum = tuple(sorted(list(numeros_em_comum)))
            
            if comb_comum not in repeticoes[quantidade_comum]:
                repeticoes[quantidade_comum][comb_comum] = set([c1, c2])
            else:
                repeticoes[quantidade_comum][comb_comum].add(c1)
                repeticoes[quantidade_comum][comb_comum].add(c2)

# 3. Exibir os Resultados Finais
print("\n" + "=" * 60)
print("ESTATÍSTICAS DE REPETIÇÃO EXCLUSIVA DE JOGOS")
print("=" * 60)

nomes = {6: "SENAS", 5: "QUINAS", 4: "QUADRAS", 3: "TERNOS", 2: "DUQUES"}

for faixa in [6, 5, 4, 3, 2]:
    total_combinacoes_distintas = len(repeticoes[faixa])
    
    print(f"\n▶ {nomes[faixa]} (Compartilhamento exato de {faixa} números):")
    print(f"  Total de grupos de {faixa} números que já se repetiram: {total_combinacoes_distintas}")
    
    if total_combinacoes_distintas > 0:
        # Criar um DataFrame temporário para ordenar e mostrar os exemplos mais frequentes
        dados_faixa = []
        for comb, concs in repeticoes[faixa].items():
            dados_faixa.append({
                'Combinação': comb,
                'Vezes': len(concs),
                'Concursos': sorted(list(concs))
            })
        
        df_faixa = pd.DataFrame(dados_faixa).sort_values(by='Vezes', ascending=False)
        print("  Top 3 combinações mais repetidas nessa faixa:")
        print(df_faixa.head(3).to_string(index=False))
    else:
        print("  Nenhuma ocorrência registrada até hoje.")

print("\n" + "=" * 60)