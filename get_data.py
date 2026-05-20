import requests
import pandas as pd
import io

# URL do arquivo Excel da Mega-Sena
excel_url = "https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"

# Faz o download do arquivo
response = requests.get(excel_url)
response.raise_for_status()  # Levanta um erro para status HTTP ruins

# Lê o conteúdo do arquivo em um DataFrame do pandas
df_mega_sena = pd.read_excel(io.BytesIO(response.content))

print("Arquivo Excel baixado e carregado no DataFrame 'df_mega_sena'.")
# Exibe as primeiras 5 linhas do DataFrame para verificação
print(df_mega_sena.head())