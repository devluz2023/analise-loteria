import pytest
import pandas as pd
from src.utils import LoteriaUtils

# Fixture para criar o ambiente de teste uma única vez para todos os testes
@pytest.fixture
def utils():
    # Cria um DataFrame de exemplo focado em testar a lógica da classe
    data = {
        'Concurso': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Bola1': [10, 10, 5, 10, 8, 10, 10, 10, 10],
        'Bola2': [2, 3, 6, 11, 9, 11, 12, 13, 14],
        'Bola3': [3, 4, 7, 12, 10, 12, 13, 14, 15],
        'Bola4': [4, 5, 8, 13, 11, 13, 14, 15, 16],
        'Bola5': [5, 6, 9, 14, 12, 14, 15, 16, 17],
        'Bola6': [6, 7, 10, 15, 13, 15, 16, 17, 18]
    }
    df_test = pd.DataFrame(data)
    # Retorna uma instância da classe já inicializada com o df de teste
    return LoteriaUtils(df_test)

def test_calcular_indice(utils):
    assert utils.calcular_indice([1, 2, 3, 4, 5, 6]) == 0
    assert utils.calcular_indice([6, 5, 4, 3, 2, 1]) == 0

def test_maior_sequencia_circular(utils):
    assert utils.maior_sequencia_circular([1, 2, 3, 10, 11]) == 3
    assert utils.maior_sequencia_circular([58, 59, 60, 1, 2]) == 5
    assert utils.maior_sequencia_circular([1, 3, 5, 7, 9, 11]) == 1

def test_contar_pares(utils):
    assert utils.contar_pares([2, 4, 6, 1, 3, 5]) == 3
    assert utils.contar_pares([1, 3, 5, 7, 9, 11]) == 0
    assert utils.contar_pares([2, 4, 6, 8, 10, 12]) == 6

def test_calcular_acertos_proximo(utils):
    # Teste de interseção baseado nos dados da fixture
    # Concurso 1: {10, 2, 3, 4, 5, 6}
    # Concurso 2: {10, 3, 4, 5, 6, 7}
    # Interseção: 10, 3, 4, 5, 6 (5 acertos)
    assert utils.calcular_acertos_proximo(0) == 5
    
    # Teste de limite
    assert utils.calcular_acertos_proximo(8) == 0

def test_calcular_maximo_repeticao(utils):
    # Com base nos dados da fixture, a dezena 10 repete-se 9 vezes seguidas
    assert utils.calcular_maximo_repeticao(10) == 9
    # Dezena inexistente
    assert utils.calcular_maximo_repeticao(99) == 0