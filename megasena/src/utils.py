from model import colunas_bolas, TOTAL_POSSIVEL
import math

class LoteriaUtils:
    def __init__(self, df_sorteios):
        """
        Inicializa o utilitário com o DataFrame de histórico.
        """
        self.df = df_sorteios
        self.colunas = colunas_bolas
        self.k = len(colunas_bolas)
        self.total_possivel = TOTAL_POSSIVEL

    def contar_pares(self, comb_list):
        return len([n for n in comb_list if int(n) % 2 == 0])

    def calcular_acertos_proximo(self, row_idx):
        if row_idx >= len(self.df) - 1:
            return 0
        atual = set([int(self.df.loc[row_idx, c]) for c in self.colunas])
        proximo = set([int(self.df.loc[row_idx + 1, c]) for c in self.colunas])
        return len(atual.intersection(proximo))

    def calcular_indice(self, comb_list):
        comb_list = sorted(comb_list)
        index = 0
        start = 1
        for i in range(self.k):
            for val in range(start, comb_list[i]):
                index += math.comb(60 - val, self.k - 1 - i)
            start = comb_list[i] + 1
        return index

    def calcular_distancia_circular(self, idx_atual, idx_proximo):
        dist = abs(idx_proximo - idx_atual)
        return min(dist, self.total_possivel - dist)

    def maior_sequencia_circular(self, comb_list):
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
        if len(grupos) > 1 and 1 in grupos[0] and 60 in grupos[-1]:
            grupos[0] = grupos[-1] + grupos[0]; grupos.pop()
        return max([len(g) for g in grupos])

    def calcular_maximo_repeticao(self, dezena):
        df_cron = self.df.sort_values(by='Concurso')
        max_streak = 0
        contador = 0
        for _, row in df_cron.iterrows():
            sorteados = [int(row[c]) for c in self.colunas]
            if dezena in sorteados:
                contador += 1
            else:
                if contador > max_streak:
                    max_streak = contador
                contador = 0
        return max(max_streak, contador)