import random
from enum import Enum


class Action(Enum):
    HIT = 0
    STICK = 1


class State:
    def __init__(self, dealer=0, player=0, deal=False):

        if deal:
            self.dealer = draw_black()
            self.player = draw_black()
        else:
            self.dealer = dealer
            self.player = player

    def copy(self):
        return State(self.dealer, self.player)

    def deal_player(self):
        self.player += draw()

    def deal_dealer(self):
        self.dealer += draw()

    def set_terminal(self):
        self.dealer = 0
        self.player = 0

    def terminal(self):
        return self.get_state() == (0, 0)

    def get_state(self) -> tuple:
        return self.dealer, self.player


def draw() -> int:
    val = random.randint(1, 10)
    black = random.random() > 1 / 3
    return val if black else -val


def draw_black() -> int:
    return random.randint(1, 10)


def bust(val) -> bool:
    return val < 1 or val > 21


def calc_terminal_reward(state: State) -> int:
    if state.player > state.dealer:
        return 1
    elif state.player < state.dealer:
        return -1
    else:
        return 0


def step(state: State, action: Action) -> tuple:
    next_state = state.copy()
    reward = 0

    if action == Action.HIT:
        next_state.deal_player()
        if bust(next_state.player):
            reward = -1
            next_state.set_terminal()

    elif action == Action.STICK:
        while next_state.dealer in range(1, 17):
            next_state.deal_dealer()

        if bust(next_state.dealer):
            reward = 1
        else:
            reward = calc_terminal_reward(next_state)

        next_state.set_terminal()

    return next_state, reward


if __name__ == '__main__':
    print("game file")
