from metaflow import FlowSpec, step

# Importando suas funções já existentes
from get_data import baixar_dados_quina
from matriz_frequencia_quina import gerar_matriz_quina
# Supondo que você ajustou seus scripts para exportar funções
# e não rodar o código no nível global
from get_data import baixar_dados_quina

class QuinaPipelineFlow(FlowSpec):

    @step
    def start(self):
        """1. Baixa os dados"""
        self.df = baixar_dados_quina()
        self.next(self.generate_matr)

    @step
    def generate_matrix(self):
        """2. Gera a Matriz de Frequência"""
        self.df_matriz = gerar_matriz_quina(self.df)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """3. Prepara as colunas (Indice, Distancia, etc)"""
        # Aqui você chama a lógica que estava no seu feature.py
        self.df_final = processar_features(self.df, self.df_matriz)
        self.next(self.train_lstm)

    @step
    def train_lstm(self):
        """4. Treina o Modelo"""
        resultado = treinar_modelo_lstm(self.df_final)
        print(f"Modelo treinado com sucesso. Resultado: {resultado}")
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline Quina concluído!")

if __name__ == '__main__':
    QuinaPipelineFlow()