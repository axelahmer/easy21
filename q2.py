from game import step, State
import matplotlib.pyplot as plt
import numpy as np
from q1 import monte_carlo_online_control
from utils import state_action_map, state_map, get_e_greedy_action
from utils import ACTIONS, DEALER_RANGE, PLAYER_RANGE

MC_Q_STAR_EPISODES = 100_000
NUM_EPISODES = 1_000

def calc_mse(Q, Qstar):
    return sum([(Q[sa] - Qstar[sa]) ** 2 for sa in Qstar.keys()]) / len(Qstar.keys())


def sarsa(lamb: int, num_episodes: int, Qstar, record=False):
    Q = state_action_map(plus=True)
    N = state_action_map()
    N_s = state_map(plus=True)
    mses = []
    for k in range(num_episodes):
        E = state_action_map()
        s = State(deal=True)
        a = get_e_greedy_action(Q, N_s, s)
        while not s.terminal():
            N_s[s.get_state()] += 1
            N[s.get_state(), a] += 1
            s_dash, r = step(s, a)
            a_dash = get_e_greedy_action(Q, N_s, s_dash)
            delta = r + Q[s_dash.get_state(), a_dash] - Q[s.get_state(), a]
            E[s.get_state(), a] += 1

            for d in DEALER_RANGE:
                for p in PLAYER_RANGE:
                    for action in ACTIONS:
                        Q[(d, p), action] += (1 / (N[(d, p), action] + 1e-9)) * delta * E[(d, p), action]
                        E[(d, p), action] *= lamb
            s = s_dash
            a = a_dash
        if record:
            mses.append(calc_mse(Q, Qstar))
    return Q, mses


def plot_mse_lambdas(num_lambdas, num_episodes, Qstar):
    lambdas = np.linspace(0, 1, num_lambdas)
    mses = []
    for lamb in lambdas:
        Q, _ = sarsa(lamb, num_episodes, Qstar)
        mses.append(calc_mse(Q, Qstar))
    plt.scatter(lambdas, mses)
    plt.title(f'Mean Square Error vs. Lambda (num eps: {num_episodes})')
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.show()


def plot_mse_over_episodes(lambdas: list, num_episodes: int, Qstar):
    mses_per_lambda = []
    for lamb in lambdas:
        _, mses = sarsa(lamb, num_episodes, Qstar, record=True)
        mses_per_lambda.append(mses)

    for mses, l in zip(mses_per_lambda, lambdas):
        plt.plot(range(num_episodes), mses, label=f'lambda = {l}')

    plt.title("Mean Square Error vs. Episode")
    plt.legend(loc='upper right')
    plt.xlabel("episode")
    plt.ylabel("MSE")
    plt.show()


if __name__ == '__main__':
    Qstar = monte_carlo_online_control(MC_Q_STAR_EPISODES)
    plot_mse_lambdas(11, NUM_EPISODES, Qstar)
    plot_mse_over_episodes([0, 1], NUM_EPISODES, Qstar)
