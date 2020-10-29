from game import step, State, Action
import random
import matplotlib.pyplot as plt
import numpy as np

ACTIONS = [Action.HIT, Action.STICK]
DEALER_RANGE = range(1, 11)
PLAYER_RANGE = range(1, 22)


def plot_values(Q):
    D = []
    P = []
    MQ = []
    for d in DEALER_RANGE:
        for p in PLAYER_RANGE:
            max_q = -1e9
            for a in ACTIONS:
                q = Q[(d, p), a]
                if q > max_q:
                    max_q = q
            D.append(d)
            P.append(p)
            MQ.append(max_q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("dealer card")
    ax.set_ylabel("player sum")
    ax.set_zlabel("V*")
    ax.plot_trisurf(D, P, MQ)

    plt.show()


def state_action_map(plus=False):
    sa_map = dict()
    if plus:
        for a in ACTIONS:
            sa_map[(0, 0), a] = 0.0

    for d in DEALER_RANGE:
        for p in PLAYER_RANGE:
            for a in ACTIONS:
                sa_map[(d, p), a] = 0.0
    return sa_map


def state_map(plus=False):
    s_map = dict()
    if plus:
        s_map[(0, 0)] = 0

    for d in DEALER_RANGE:
        for p in PLAYER_RANGE:
            s_map[d, p] = 0
    return s_map


def sample_episode(pi):
    history = []
    s = State(deal=True)

    while not s.terminal():
        a = pi[s.get_state()]
        # rewards do not need to be appended to history as rewards are only *rewarded* when entering the terminal state.
        history.append([s.get_state(), a])
        s, r = step(s, a)

    return history, r


def get_e_greedy_action(Q: dict, N: dict, state: State):
    epsilon = 100 / (100 + N[state.get_state()])
    chosen_action = None
    if np.random.uniform() > epsilon:
        max_q = -1e9
        for a in ACTIONS:
            q = Q[state.get_state(), a]
            if q > max_q:
                max_q = q
                chosen_action = a
    else:
        chosen_action = random.choice(ACTIONS)
    return chosen_action


def e_greedy(Q: dict, N: dict):
    policy = dict()

    for d in DEALER_RANGE:
        for p in PLAYER_RANGE:
            epsilon = 100 / (100 + N[(d, p)])
            chosen_action = None
            if np.random.uniform() > epsilon:
                max_q = -1e9
                for a in ACTIONS:
                    q = Q[(d, p), a]
                    if q > max_q:
                        max_q = q
                        chosen_action = a
            else:
                chosen_action = random.choice(ACTIONS)
            policy[d, p] = chosen_action
    return policy
