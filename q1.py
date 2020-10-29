from utils import plot_values, state_action_map, state_map, sample_episode, e_greedy

NUM_EPISODES = 100_000


def monte_carlo_online_control(num_episodes=NUM_EPISODES):
    # init Q(s,a)=0, N(s,a)=0 for every s a
    Q = state_action_map()
    N = state_action_map()
    N_s = state_map()

    for k in range(num_episodes):
        if k % 1000 == 0:
            print(f'{k} / {num_episodes}')
        pi = e_greedy(Q, N_s)
        episode, reward = sample_episode(pi)
        explored = set()
        for s, a in episode:
            if s not in explored:
                explored.add(s)
                N[s, a] = N[s, a] + 1
                N_s[s] = N_s[s] + 1
                Q[s, a] = Q[s, a] + (1 / N[s, a]) * (reward - Q[s, a])
            # print(f'{s}{a} -> {Q[s, a]}')

    return Q


if __name__ == '__main__':
    Q_vals = monte_carlo_online_control()
    plot_values(Q_vals)
