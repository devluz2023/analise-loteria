import numpy as np
import pandas as pd
from scipy import stats

class AnaliseEstatisticaLotofacil:
    def __init__(self, df_historico):
        # 1. Ajuste: A Lotofácil tem 15 colunas de dezenas (0 a 14)
        df_dezenas = df_historico.iloc[:, :15] 
        
        dezenas_limpas = df_dezenas.apply(pd.to_numeric, errors='coerce').values.flatten()
        self.dezenas = dezenas_limpas[~np.isnan(dezenas_limpas)].astype(int)
        
        self.total_sorteios = len(df_historico)
    
    def normalizar_log(self, frequencias):
        return np.log1p(frequencias)
    
    def validar_hipotese(self):
        # Ajuste: A Lotofácil vai de 1 a 25
        freq_contagem = pd.Series(self.dezenas).value_counts().reindex(range(1, 26), fill_value=0).sort_index()
        
        # P_esperado: Na Lotofácil, a probabilidade é 15/25 = 0.6
        p_esperado = 0.6 
        
        p_observado = freq_contagem / self.total_sorteios
        
        # Z-Test
        n = self.total_sorteios
        std_erro = np.sqrt(p_esperado * (1 - p_esperado) / n)
        z_scores = (p_observado - p_esperado) / std_erro
        
        return z_scores, p_observado

    def encontrar_racional(self):
        z_scores, p_obs = self.validar_hipotese()
        norm_freq = self.normalizar_log(p_obs)
        
        racional = pd.DataFrame({
            'Z_Score': z_scores,
            'Freq_Norm': norm_freq,
            'Racional': np.where(z_scores > 1.96, 'Tendencia_Alta', 
                                np.where(z_scores < -1.96, 'Tendencia_Baixa', 'Equilibrio'))
        }, index=range(1, 26))
        return racional