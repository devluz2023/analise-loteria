import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from get_data import df_mega_sena

# === IMPORTAÇÕES DA SUA LÓGICA OFICIAL ===
from indice import encontrar_posicao_itertools

print("Inicializando Motor Geométrico do Hipercubo Espacial (6D)...")
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
df_cronologico = df_mega_sena.sort_values(by='Concurso').copy()

# =============================================================
# 1. CÁLCULO DO VETOR CENTRO DE GRAVIDADE E FRONTEIRAS DO CUBO
# =============================================================
# Converte o histórico em uma matriz de vetores 6D
matriz_jogos_6d = df_cronologico[colunas_bolas].values.astype(np.float32)

# Vetor Centro de Gravidade Histórico (Média de cada dimensão/bola)
vetor_centro_mu = np.mean(matriz_jogos_6d, axis=0)

# Calcula as distâncias euclidianas de todos os jogos reais até o centro mu
# d = sqrt( sum( (bola_i - mu_i)^2 ) )
distancias_historicas = np.sqrt(np.sum((matriz_jogos_6d - vetor_centro_mu) ** 2, axis=1))

# Definição Não Linear da Casca Externa (Concentração de Medida)
# Usamos a média das distâncias e o desvio padrão para achar as "Faces Externas"
media_distancia = np.mean(distancias_historicas)
desvio_distancia = np.std(distancias_historicas)

# Limites integrados da casca do hipercubo (Onde a densidade é máxima)
RAIO_MIN_CASCA = media_distancia - (0.5 * desvio_distancia)
RAIO_MAX_CASCA = media_distancia + (1.2 * desvio_distancia)

print(f"-> Vetor Centro de Gravidade 6D (μ): {np.round(vetor_centro_mu, 2)}")
print(f"-> Fronteira da Casca Externa: Raio Mínimo = {RAIO_MIN_CASCA:.2f} | Raio Máximo = {RAIO_MAX_CASCA:.2f}\n")


# =============================================================
# 2. SELEÇÃO DA BASE DE ELITE VIA ADERÊNCIA ÀS FACES EXTERNAS
# =============================================================
# Vamos reconstruir a força das dezenas usando a proximidade com as faces do hipercubo
df_ausencia = pd.read_excel("matriz_ausencia_completa.xlsx")
colunas_jejum = [col for col in df_ausencia.columns if str(col).isdigit()]

distribuicao_global_jejuns = {int(col): 0 for col in colunas_jejum}
for index, row in df_ausencia.iterrows():
    for col in colunas_jejum:
        distribuicao_global_jejuns[int(col)] += int(row[col])

total_sequencias = sum(distribuicao_global_jejuns.values())
prob_prior_jejum = {t: (qtd / total_sequencias) for t, qtd in distribuicao_global_jejuns.items()}

current_ausencia = {d: 0 for d in range(1, 61)}
for index, row in df_cronologico.iterrows():
    sorteados = set([row[col] for col in colunas_bolas])
    for d in range(1, 61):
        if d in sorteados: current_ausencia[d] = 0
        else: current_ausencia[d] += 1

# Filtra um subconjunto de dezenas com base nas janelas estocásticas de Bayes para alimentar o gerador
scores_bayes = []
for d in range(1, 61):
    jejum_futuro = current_ausencia[d] + 1
    scores_bayes.append((d, prob_prior_jejum.get(jejum_futuro, 0.0001)))
    
scores_bayes_ordenados = sorted(scores_bayes, key=lambda x: x[1], reverse=True)
universo_dezenas_cubo = [item[0] for item in scores_bayes_ordenados[:28]]


# =============================================================
# 3. FILTRAGEM GEOMÉTRICA DE COORDENADAS (INTEGRAÇÃO E DERIVADA)
# =============================================================
jogos_vetoriais_aprovados = []
tentativas_espaciais = 0

while len(jogos_vetoriais_aprovados) < 5 and tentativas_espaciais < 60000:
    tentativas_espaciais += 1
    
    # Sorteia coordenadas dentro do subespaço de dezenas selecionadas
    palpite_vetor = np.array(sorted(random.sample(universo_dezenas_cubo, 6)), dtype=np.float32)
    
    # --- CÁLCULO DA DISTÂNCIA VETORIAL 6D ---
    # Mede a distância do palpite candidato até o centro de gravidade mu
    distancia_palpite = np.sqrt(np.sum((palpite_vetor - vetor_centro_mu) ** 2))
    
    # --- FILTRO GEOMÉTRICO 1: Restrição da Casca Externa (Integral de Volume) ---
    # Se o jogo estiver muito perto do centro (raio < min) ou muito longe (raio > max), é descartado
    if not (RAIO_MIN_CASCA <= distancia_palpite <= RAIO_MAX_CASCA):
        continue
        
    # --- FILTRO GEOMÉTRICO 2: Derivada/Gradiente Direcional (Taxa de Variação) ---
    # Calculamos a derivada da distância em relação aos eixos ordenados.
    # Em termos práticos de código, a variação (diferença) entre os passos do vetor 
    # não pode apresentar picos abruptos (saltos gigantescos entre dezenas vizinhas)
    gradiente_passos = np.diff(palpite_vetor)
    if np.max(gradiente_passos) > 24 or np.min(gradiente_passos) < 1:
        continue
        
    # --- FILTRO GEOMÉTRICO 3: Balanço Estatístico de Paridade ---
    if sum(1 for n in palpite_vetor if n % 2 == 0) not in [2, 3, 4]:
        continue
        
    # Evita duplicar jogos e converte float para int para saída limpa
    jogo_convertido = [int(num) for num in palpite_vetor]
    if jogo_convertido not in jogos_vetoriais_aprovados:
        jogos_vetoriais_aprovados.append(jogo_convertido)


# =============================================================
# 4. EMISSÃO DOS PALPITES GEOMÉTRICOS ÓTIMOS
# =============================================================
print("=" * 75)
print("             JOGOS GERADOS VIA GEOMETRIA DE CASCA DE HIPERCUBO")
print("=" * 75)
print(f"Coordenadas escaneadas no subespaço vetorial: {tentativas_espaciais}")
print("-" * 75)

for i, jogo in enumerate(jogos_vetoriais_aprovados, 1):
    jogo_formatado = ", ".join(f"{num:02d}" for num in jogo)
    linha_pandas = pd.Series(data=jogo, index=colunas_bolas)
    
    # Calcula a distância exata que esse bilhete ficou da casca
    dist_centro = np.sqrt(np.sum((np.array(jogo) - vetor_centro_mu) ** 2))
    
    indice_calculado = encontrar_posicao_itertools(linha_pandas)
    indice_br = f"{indice_calculado:,}".replace(",", ".")
    
    print(f"Jogo {i}: [ {jogo_formatado} ] ➔ Índice Real: {indice_br} | Distância Vetorial: {dist_centro:.2f}")

print("=" * 75)