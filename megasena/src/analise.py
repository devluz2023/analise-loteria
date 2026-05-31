import numpy as np
import pandas as pd
from scipy import stats

class AnaliseEstatisticaMega:
    def __init__(self, df_historico):
        # 1. Seleciona as colunas de dezenas e força a conversão para numérico
        # Isto resolve o conflito entre int e str
        df_dezenas = df_historico.iloc[:, :6]
        
        # Converte para numérico, transformando erros em NaN
        dezenas_limpas = df_dezenas.apply(pd.to_numeric, errors='coerce').values.flatten()
        
        # Remove eventuais valores NaN (limpeza) e converte para int
        self.dezenas = dezenas_limpas[~np.isnan(dezenas_limpas)].astype(int)
        
        self.total_sorteios = len(df_historico)
        self.janela = 10  # A cada 10 jogos
    
    def normalizar_log(self, frequencias):
        # Logaritmo para normalizar a escala
        return np.log1p(frequencias)
    
    def validar_hipotese(self):
        """
        H0: A dezena tem probabilidade esperada de 1/10 (10% na janela de 10 jogos).
        Calcula o Z-Score para ver se a dezena está fora do esperado.
        """
        # Agora o sort_index funcionará porque self.dezenas são inteiros puros
        freq_contagem = pd.Series(self.dezenas).value_counts().sort_index()
        p_esperado = 1 / self.janela
        
        # Probabilidade real observada
        p_observado = freq_contagem / self.total_sorteios
        
        # Z-Test: (p_obs - p_esp) / sqrt(p_esp * (1 - p_esp) / n)
        n = self.total_sorteios
        std_erro = np.sqrt(p_esperado * (1 - p_esperado) / n)
        z_scores = (p_observado - p_esperado) / std_erro
        
        return z_scores, p_observado

    def encontrar_racional(self):
        z_scores, p_obs = self.validar_hipotese()
        norm_freq = self.normalizar_log(p_obs)
        
        # Racional Estatístico
        # Z > 1.96 (Significativo: dezena "quente")
        # Z < -1.96 (Significativo: dezena "fria")
        racional = pd.DataFrame({
            'Z_Score': z_scores,
            'Freq_Norm': norm_freq,
            'Racional': np.where(z_scores > 1.96, 'Tendencia_Alta', 
                                np.where(z_scores < -1.96, 'Tendencia_Baixa', 'Equilibrio'))
        })
        return racional


