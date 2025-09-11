from lib.custom import automata # Importação atualizada
import unittest
import numpy as np

class TestAutomata(unittest.TestCase): # Classe renomeada para maior clareza

    def test_still(self) -> None:
        """2x2 block"""

        A: np.ndarray = np.zeros((10,10))
        A[1:3,1:3] = 1
        B: np.ndarray = automata(A) # Use autômatos
        assert (A == B).all()

    def test_oscillator(self) -> None: # Erro de digitação corrigido: scillator -> oscillator
        """blinker"""
        A: np.ndarray = np.zeros((10,10))
        A[1:4,1] = 1
        B: np.ndarray = automata(A) # Use autômatos
        assert (B[2, 0:3] == 1).all()

        B = automata(B) # Use autômatos
        assert (A == B).all()

    def test_evolution(self) -> None:
        """test that something changes"""
        m, n = 10, 10
        A: np.ndarray = np.random.random(m*n).reshape((m, n)).round()
        B: np.ndarray = automata(A) # Use autômatos
        assert (B != A).any()



if __name__ == '__main__':
    unittest.main()
