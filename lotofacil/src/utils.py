import math
from model import colunas_bolas, TOTAL_POSSIVEL

class LoteriaUtils:
    def __init__(self, df_sorteios):
        """
        Inicializa o utilitário para Lotofácil.
        """
        self.df = df_sorteios
        self.colunas = colunas_bolas
        self.k = len(colunas_bolas) # k = 15
        self.total_possivel = TOTAL_POSSIVEL
        self.universo = 25 # Ajuste para Lotofácil

    def contar_pares(self, comb_list):
        return len([n for n in comb_list if int(n) % 2 == 0])

    def calcular_acertos_proximo(self, row_idx):
        if row_idx >= len(self.df) - 1:
            return 0
        atual = set([int(self.df.loc[row_idx, c]) for c in self.colunas])
        proximo = set([int(self.df.loc[row_idx + 1, c]) for c in self.colunas])
        return len(atual.intersection(proximo))

    def calcular_indice(self, comb_list):
        """
        Calcula o índice combinatório para o universo de 25 bolas e k=15.
        """
        comb_list = sorted([int(n) for n in comb_list])
        index = 0
        start = 1
        for i in range(self.k):
            for val in range(start, comb_list[i]):
                # Ajuste para universo de 25
                index += math.comb(self.universo - val, self.k - 1 - i)
            start = comb_list[i] + 1
        return index

    def calcular_distancia_circular(self, idx_atual, idx_proximo):
        # A Lotofácil não tem "distância circular" real, 
        # mas mantemos para compatibilidade com o teu pipeline.
        dist = abs(idx_proximo - idx_atual)
        return min(dist, self.total_possivel - dist)

    def maior_sequencia_circular(self, comb_list):
        """
        Calcula a maior sequência de números seguidos na Lotofácil.
        """
        comb = sorted([int(n) for n in comb_list])
        grupos = []
        i = 0
        while i < len(comb):
            g = [comb[i]]
            while i + 1 < len(comb) and comb[i+1] == comb[i] + 1:
                g.append(comb[i+1])
                i += 1
            grupos.append(g)
            i += 1
        # Não aplicamos circularidade de 25 -> 1 na Lotofácil por padrão
        return max([len(g) for g in grupos]) if grupos else 0

    def calcular_maximo_repeticao(self, dezena):
        """
        Calcula a maior sequência de vezes que uma dezena saiu seguidamente.
        """
        df_cron = self.df.sort_values(by='Concurso')
        max_streak = 0
        contador = 0
        for _, row in df_cron.iterrows():
            # A Lotofácil tem 15 colunas, garante que a busca engloba todas
            sorteados = [int(row[c]) for c in self.colunas]
            if dezena in sorteados:
                contador += 1
            else:
                if contador > max_streak:
                    max_streak = contador
                contador = 0
        return max(max_streak, contador)