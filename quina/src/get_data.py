import numpy as np
import pandas as pd
import requests
import io
import math
import os
from collections import Counter


# === CONFIGURAÇÃO ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TOTAL_POSSIVEL = math.comb(80, 5)


# === 1. DOWNLOAD E PREPARAÇÃO ===
print("Baixando dados oficiais e processando...")
url_quina = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Quina"
response = requests.get(url_quina, verify=False)
df = pd.read_excel(io.BytesIO(response.content))
colunas_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5']
df = df.sort_values(by='Concurso').reset_index(drop=True)

# Gera a matriz de estatísticas completa