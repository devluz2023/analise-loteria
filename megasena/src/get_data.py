import numpy as np
import pandas as pd
import requests
import io
import os
from collections import Counter


# === CONFIGURAÇÃO ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# === 1. DOWNLOAD E PREPARAÇÃO ===
print("Baixando dados oficiais e processando...")
excel_url = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
response = requests.get(excel_url, verify=False)
df = pd.read_excel(io.BytesIO(response.content))

df = df.sort_values(by='Concurso').reset_index(drop=True)

# Gera a matriz de estatísticas completa