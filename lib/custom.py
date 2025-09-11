import numpy as np
# Removido: from .lib import fft_convolve2d - Não usa mais convolução FFT personalizada
from scipy.signal import convolve2d # Use a convolução do SciPy's 
from typing import Tuple, List

def parse_rule(rule: str) -> Tuple[List[int], List[int]]:
    """
    parses B/S rule strings
    """
    born = [int(i) for i in rule.split('/')[0][1:]]
    survive = [int(i) for i in rule.split('/')[1][1:]]
    return born, survive

# MODIFICADO: Argumento 'limite' adicionado
def automata(state: np.ndarray, rule: str = 'B3/S23', boundary: str = 'wrap') -> np.ndarray:
    """
    General cellular automata state transition function.

    Args:
        state (np.ndarray): The current board state (0s and 1s).
        rule (str): The ruleset string (e.g., 'B3/S23').
        boundary (str): Boundary condition for convolution.
                        'wrap' for periodic (toroidal),
                        'fill' for zero-padding.
                        Defaults to 'wrap'.

    Returns:
        np.ndarray: The next board state.
    """
    # Defina o kernel padrão de vizinhança de Moore 3x3 (excluindo o centro)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Analisar a sequência de regras
    born_rules, survive_rules = parse_rule(rule)

    # Executar convolução 2D usando SciPy
    # O modo 'same' garante que a saída tenha o mesmo formato do estado de entrada
    # O modo 'boundary' controla como as arestas são manipuladas ('wrap' ou 'fill')
    neighbor_sum = convolve2d(state, kernel, mode='same', boundary=boundary, fillvalue=0)

    # Aplique as regras
    next_state: np.ndarray = np.zeros_like(state)

    # Aplicar regras de sobrevivência (a célula atual está viva)
    for s_rule in survive_rules:
        next_state[(state == 1) & (neighbor_sum == s_rule)] = 1

    # Aplicar regras de nascimento (a célula atual está morta)
    for b_rule in born_rules:
        next_state[(state == 0) & (neighbor_sum == b_rule)] = 1

    return next_state

# Exemplo de uso (opcional, para teste)
if __name__ == '__main__':
    # Exemplo: Jogo da Vida de Conway
    initial_state = np.zeros((10, 10))
    initial_state[4:7, 4:7] = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) # Blinker
    print("Initial State (Blinker):")
    print(initial_state)

    print("\nNext State (Wrap Boundary):")
    next_state_wrap = automata(initial_state, rule='B3/S23', boundary='wrap')
    print(next_state_wrap)

    print("\nNext State (Fill Boundary):")
    next_state_fill = automata(initial_state, rule='B3/S23', boundary='fill')
    print(next_state_fill)

    # Exemplo: Replicador
    initial_state_rep = np.zeros((5, 5))
    initial_state_rep[2, 1:4] = 1
    print("\nInitial State (Replicator Line):")
    print(initial_state_rep)
    print("\nNext State (Replicator, Wrap):")
    next_state_rep = automata(initial_state_rep, rule='B1357/S1357', boundary='wrap')
    print(next_state_rep)
