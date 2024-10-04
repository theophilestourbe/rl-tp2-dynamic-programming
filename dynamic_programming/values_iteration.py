import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    old_values = None
    eps = 1e-4
    iter = 0
    while (
        old_values is None or np.abs(values - old_values).sum() > eps or iter > max_iter
    ):
        old_values = np.copy(values)
        for s in range(mdp.observation_space.n):
            mdp.reset_state(s)
            am = -100
            for a in range(mdp.action_space.n):
                S = 0
                for sp in range(mdp.observation_space.n):
                    n, r, f, _ = mdp.step(a)
                    if n != sp:
                        continue
                    S += r + gamma * old_values[sp]
                am = max(am, S)
            values[s] = am
        iter += 1
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    old_values = None
    iter = 0
    delta = 0
    while iter < max_iter or delta > theta:
        env.reset()
        old_values = np.copy(values)
        delta = 0
        for row in range(env.observation_space.spaces[0].n):
            for col in range(env.observation_space.spaces[1].n):
                env.set_state(row, col)
                delta += value_iteration_per_state(
                    env, values, gamma, old_values, delta
                )
        iter += 1
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    old_values = None
    iter = 0
    delta = 0
    while iter < max_iter or delta > theta:
        env.reset()
        old_values = np.copy(values)
        delta = 0
        for row in range(env.observation_space.spaces[0].n):
            for col in range(env.observation_space.spaces[1].n):
                env.set_state(row, col)
                delta += value_iteration_per_state(
                    env, values, gamma, old_values, delta
                )
        iter += 1
    return values
