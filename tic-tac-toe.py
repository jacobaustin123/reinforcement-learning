"""

Random, fixed opponent (moves fixed, but randomly chosen the first time through)

Trained agent character, using a dict lookup of tuple of tuple of board positions. i.e. let player be o and move first.

Check that neither player has won, look up board position in defaultdict (or dict.get), assign .5 to it if it doesn't exit, then iterate over the board and look up all possible board states in dict. Sometimes, choose an exploratory move (at random, or from the top choices), or else choose the best move. After each move, update previous state value.

d = defaultdict(lambda:.5)


1. Check if game is over. If so, set game_state[state] to 0.

2. Check if exploratory move (random number less than alpha).

3. If not exploratory move, list all open positions, look up values for each. Check if move wins or draws the game, assign zero or 1

If exploratory move, select random open position and play it.

Otherwise pick the best one, then backup value i, i.e. make value[state] more like value[current_state]

update self.state,

"""

from collections import defaultdict
import random
import re
import numpy as np
import matplotlib.pyplot as plt

class Board:
    def __init__(self, custom_state = None, verbose = False):
        self.state = custom_state if custom_state else [['', '', ''], ['','',''],['','','']]
        self.gameover = False
        self.xwins = 0
        self.owins = 0
        self.draws = 0
        self._num_games = 0
        self.verbose = verbose
        self.history = []
        self.last = np.array([0, 0, 0])
        self.interval = 20

    def __getitem__(self, n):
        return self.state[n]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __str__(self):
        flat = [x for sublist in self.state for x in sublist]
        return "----------\n| {0} | {1} | {2} |\n|---------\n| {3} | {4} | {5} |\n|---------\n| {6} | {7} | {8} |\n----------".format(*flat)

    def num_games(self):
        self._num_games=self.xwins + self.owins + self.draws
        return self._num_games

    def update_history(self):
        if self.num_games() % self.interval == 0:
            self.temp = np.array([self.owins - self.last[0], self.xwins - self.last[1], self.draws - self.last[2]]) / self.interval
            self.history.append(self.temp)
            self.last = [self.owins, self.xwins, self.draws]


    def getstate(self):
        return tuple(tuple(a) for a in self.state)

    def reset(self):
        self.state = [['', '', ''], ['','',''], ['','','']]
        self.gameover = False

    def check_win(self, player):
        return (any(all(self.state[i][j] == player for j in range(3)) for i in range(3)) or
                any(all(self.state[i][j] == player for i in range(3)) for j in range(3)) or
                all(self.state[i][i] == player for i in range(3)) or
                all(self.state[i][2 - i] == player for i in range(3)))

    def check_draw(self):
        return all(self.state[i][j] != '' for i in range(3) for j in range(3))

    def check_gameover(self):
        if not self.gameover: #TODO remove this later
            if self.check_win('x'):
                self.xwins += 1
                self.gameover = True
                if self.verbose: print("X wins!")
                return True
            elif self.check_win('o'):
                self.owins += 1
                self.gameover = True
                if self.verbose: print("O wins!")
                return True
            elif self.check_draw():
                self.draws +=1
                self.gameover = True
                if self.verbose: print("Draw!")
                return True
            return False
        return True

    def draw_graph(self):
        history = np.asarray(self.history)
        fig = plt.figure()
        fig.suptitle('Win, Loss, and Draw Statistics', fontsize=20)
        plots = plt.plot(np.arange(0, len(history)), history)
        plt.xlabel("step (i / {})".format(self.interval))
        plt.ylabel("rate")
        plt.legend(iter(plots), ('o win rate', 'x win rate', 'draw rate'))
        plt.savefig('training_history.png')


class Agent:
    def __init__(self, name, trainable = False, explore = 0.4, alpha = 0.8, verbose=False):
        self.name = name
        self.trainable = trainable
        self.move_table = defaultdict(lambda: .5) if trainable else defaultdict(lambda: random.random())
        self.explore = explore
        self.alpha = alpha
        self.state = (('','',''),('','',''),('','',''))
        self.explored = True
        self.verbose = verbose

    def check_end(self, board):
        if board.check_win(self.name):
            self.move_table[board.getstate()] = 1

        elif board.check_draw():
            self.move_table[board.getstate()] = .5

    def make_move(self, board, test=False):

        values = {}
        max, max_index = -1, (0, 0)
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    index = (i, j)
                    board[i][j] = self.name
                    self.check_end(board)
                    temp_value = self.move_table[board.getstate()]
                    values[index] = temp_value
                    if temp_value > max:
                        max_index = (i, j)
                        max = temp_value
                    board[i][j] = ''

        if self.verbose:
            vals = [values.get(x, -1) for x in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]]
            print("----------\n| {:.2f} | {:.2f} | {:.2f} |\n|---------\n| {:.2f} | {:.2f} | {:.2f} |\n|---------\n| {:.2f} | {:.2f} | {:.2f} |\n----------".format(*vals))

        if random.random() < self.explore and not self.verbose: # and self.trainable
            rand_index = random.randint(0, len(values.keys()) - 1)
            index = list(values.keys())[rand_index]
            board[index[0]][index[1]] = self.name
            self.explored = True

        else:
            board[max_index[0]][max_index[1]] = self.name
            self.explored = False

        if not self.explored and self.trainable:
            self.move_table[self.state] += self.alpha * (
            self.move_table[board.getstate()] - self.move_table[self.state])

        self.state = board.getstate()

class Player:
    def __init__(self, name):
        self.name = name

    def make_move(self, board):
        if not board.check_gameover():
            print(board)
            print('\nYou are {}. Please enter your next move by index, i.e. "0,0", "2, 1", "1 1", etc.'.format(self.name))
            move = input()
            moves = list(map(int, re.findall('\d+', move)))
            print(moves)
            board[moves[0]][moves[1]] = self.name

def train(agent, agent2 = None, num_iterations = 50000, interval = 10000, alpha_decay = 0.1, explore_decay = 0.1, trainable = False):
    if agent2 and not trainable:
        agent2.trainable = False

    if agent.name == 'o':
        first_agent = agent
        if agent2:
            assert agent2.name == 'x'
            second_agent = agent2
        else:
            second_agent = Agent('x')
    if agent.name == 'x':
        second_agent = agent
        if agent2:
            assert agent2.name == 'o'
            first_agent = agent2
        else:
            first_agent = Agent('o')
    board = Board()

    for i in range(num_iterations):
        while not board.gameover:
            first_agent.make_move(board)
            if board.check_gameover():
                second_agent.move_table[second_agent.state] = 0
                break
            second_agent.make_move(board)
        else:
            first_agent.move_table[first_agent.state] = 0

        board.reset()
        board.update_history()

        if interval and i % interval == 0:
            if alpha_decay:
                Player1.alpha -= alpha_decay
            if explore_decay:
                Player1.explore -= explore_decay

    print("o won %d games, x won %d games. %d games were drawn." % (board.owins, board.xwins, board.draws))

    board.draw_graph()

def play(agent):

    if agent.name == 'x':
        first = Player('o')
        second = agent
    if agent.name == 'o':
        first = agent
        second = Player('x')

    agent.trainable = False
    agent.verbose = True
    agent.explore = 0

    board = Board(verbose=True)

    while True:
        while not board.check_gameover():
            first.make_move(board)
            if board.check_gameover():
                break
            second.make_move(board)
        board.reset()
        print("Type q to quit, anything else to play again")
        if input() == 'q':
            break

    agent.verbose = False
    agent.trainable = True


if __name__ == "__main__":

    # train 'o'

    Player1 = Agent('o', trainable=True)

    train(Player1, agent2 = None, num_iterations=50000, interval=10000, alpha_decay=0.1, explore_decay=0.2)

    Player2 = Agent('x', trainable=True)

    train(Player2, agent2 = None, num_iterations=50000, interval=10000, alpha_decay=0.1, explore_decay=0.2)

    play(Player2)

    Player1.alpha = .8
    Player1.explore = .4
    Player2.alpha = .8
    Player2.explore = .4

    train(Player1, agent2 = Player2, num_iterations=50000, interval=10000, alpha_decay=0.1, explore_decay=0.2)

    play(Player1)


