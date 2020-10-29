from game import step, State, Action
import matplotlib.pyplot as plt
import numpy as np
from q1 import monte_carlo_online_control
from utils import state_action_map, state_map
from utils import ACTIONS, DEALER_RANGE, PLAYER_RANGE

MC_Q_STAR_EPISODES = 1_000
NUM_EPISODES = 1000
ALPHA = 0.01
EPSILON = 0.05

DEALER_BUCKETS = [[1, 4], [4, 7], [7, 10]]
PLAYER_BUCKETS = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
ACTION_IDS = {Action.HIT: 0, Action.STICK: 1}


def get_bucket_ids(buckets, val):
    ids = []
    for i, bucket in enumerate(buckets):
        if val in range(bucket[0], bucket[1] + 1):
            ids.append(i)
    return ids


def phi(state: State, action: Action):
    phi = np.zeros((3, 6, 2), dtype=int)
    d_ids = get_bucket_ids(DEALER_BUCKETS, state.dealer)
    p_ids = get_bucket_ids(PLAYER_BUCKETS, state.player)
    a_id = ACTION_IDS[action]

    for d in d_ids:
        for p in p_ids:
            phi[d, p, a_id] = 1

    return phi.flatten()


def q_hat(state: State, action: Action, w):
    return np.dot(phi(state, action), w)


def v_star(state: State, w):
    return max([q_hat(state, a, w) for a in ACTIONS])


def calc_mse_linear(w, Qstar):
    return sum([(q_hat(State(d, p), a, w) - Qstar[(d, p), a]) ** 2 for (d, p), a in Qstar.keys()]) / len(Qstar.keys())


def get_e_greedy_action(state: State, w):
    epsilon = EPSILON
    chosen_action = None
    if np.random.uniform() > epsilon:
        max_q = -1e9
        for action in ACTIONS:
            q = q_hat(state, action, w)
            if q > max_q:
                max_q = q
                chosen_action = action
    else:
        chosen_action = np.random.choice(ACTIONS)
    return chosen_action


def sarsa(lamb: int, num_episodes: int, Qstar, record=False):
    alpha = ALPHA
    w = np.zeros(36)
    # w = np.random.uniform(-1, 1, 36)
    Q = state_action_map(plus=True)
    N = state_action_map()
    N_s = state_map(plus=True)
    mses = []
    for k in range(num_episodes):
        E = np.zeros(36)
        s = State(deal=True)
        a = get_e_greedy_action(s, w)
        while not s.terminal():
            x = phi(s, a)
            s_dash, r = step(s, a)
            a_dash = get_e_greedy_action(s_dash, w)

            delta = r + q_hat(s_dash, a_dash, w) - q_hat(s, a, w)
            E = np.add(np.multiply(E, lamb), x)
            dw = np.multiply(E, alpha * delta)
            w += dw

            s = s_dash
            a = a_dash
        if record:
            mses.append(calc_mse_linear(w, Qstar))
    return w, mses


def plot_mse_lambdas(num_lambdas, num_episodes, Qstar):
    lambdas = np.linspace(0, 1, num_lambdas)
    mses = []
    for lamb in lambdas:
        w, _ = sarsa(lamb, num_episodes, Qstar)
        mses.append(calc_mse_linear(w, Qstar))
    plt.scatter(lambdas, mses)
    plt.title(f'Mean Square Error vs. Lambda (num eps: {num_episodes})')
    plt.xlabel("Lambda")
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
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.show()


if __name__ == '__main__':
    Qstar = monte_carlo_online_control(MC_Q_STAR_EPISODES)
    plot_mse_lambdas(11, NUM_EPISODES, Qstar)
    plot_mse_over_episodes([0, 1], NUM_EPISODES, Qstar)
