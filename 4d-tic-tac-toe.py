import numpy as np
import re
import random

n, m = 0, 0

class Board:
    def __init__(self, verbose = False):
        self.state = [['', '', '', ''], ['','','', ''], ['','','', ''], ['','','', '']]
        self.xwins = 0
        self.owins = 0
        self.draws = 0
        self._num_games = 0
        self.verbose = verbose

    def __getitem__(self, n):
        return self.state[n]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __str__(self):
        flat = [x for sublist in self.state for x in sublist]
        return "-------------\n| {0} | {1} | {2} | {3} |\n|------------\n| {4} | {5} | {6} | {7} |\n|------------\n| {8} | {9} | {10} | {11} |\n------------\n| {12} | {13} | {14} | {15} |\n------------".format(*flat)

    def num_games(self):
        self._num_games=self.xwins + self.owins + self.draws
        return self._num_games

    def getstate(self):
        return tuple(tuple(a) for a in self.state)

    def reset(self):
        self.state = [['', '', '', ''], ['','','', ''], ['','','', ''], ['','','', '']]
        self.gameover = False

    def check_win(self, player):
        return (any(all(self.state[i][j] == player for j in range(4)) for i in range(4)) or
                any(all(self.state[i][j] == player for i in range(4)) for j in range(4)) or
                all(self.state[i][i] == player for i in range(4)) or
                all(self.state[i][3 - i] == player for i in range(4)))

    def check_draw(self):
        return all(self.state[i][j] != '' for i in range(4) for j in range(4))

    def check_gameover(self):
        if self.check_win('o'):
            if self.verbose:
                print("o wins!")
            self.owins += 1
            return True
        if self.check_win('x'):
            if self.verbose:
                print("x wins!")
            self.xwins += 1
            return True
        if self.check_draw():
            if self.verbose:
                print("Draw!")
            self.draws += 1
            return True
        return False

heuristic = [[0, -.25, -.5, -.75, -1], [.25, 0, 0, 0, 0], [.5, 0, 0, 0, 0], [.75, 0, 0, 0, 0], [1, 0, 0, 0, 0]]

def evaluate(board, player, other_player):
    score = 0

    for i in range(4):
        player_score = sum(board[i][j] == player for j in range(4))
        opponent_score = sum(board[i][j] == other_player for j in range(4))
        score += heuristic[player_score][opponent_score]

    for j in range(4):
        player_score = sum(board[i][j] == player for i in range(4))
        opponent_score = sum(board[i][j] == other_player for i in range(4))
        score += heuristic[player_score][opponent_score]

    player_score = sum(board[i][i] == player for i in range(4))
    opponent_score = sum(board[i][i] == other_player for i in range(4))
    score += heuristic[player_score][opponent_score]

    player_score = sum(board[i][3 - i] == player for i in range(4))
    opponent_score = sum(board[i][3 - i] == player for i in range(4))
    score += heuristic[player_score][opponent_score]

    return score

def minimax(board, player, players, max, depth):
    global n, m

    if players[0] == player:
        other_player = players[1]
    else:
        other_player = players[0]

    if board.check_win(player):
        return 1 #random.randint(0, 9)
    if board.check_draw():
        return 0 #random.randint(0, 9)
    if board.check_win(other_player):
        return -1 #random.randint(0, 9)

    if depth == 0:
        score = evaluate(board, player, other_player)
        return score

    best_score = -2

    for i in range(4):
        for j in range(4):
            if board[i][j] == '':
                if best_score >= max:
                    n += 1
                    return best_score
                board[i][j] = player
                subscore = - minimax(board, other_player, players, - best_score, depth - 1)
                board[i][j] = ''
                if subscore > best_score:
                    best_score = subscore

    m += 1
    if m % 1000 == 0:
        print("m =", m, "n=", n)
    return best_score

def find_move(board, player, players, depth):


    if players[0] == player:
        other_player = players[1]
    else:
        other_player = players[0]

    best_score, index = -1, (0, 0)

    values = []
    for j in range(4):
        for i in range(4):
            if board[i][j] == '':
                board[i][j] = player
                subscore = - minimax(board, other_player, players, - best_score, depth - 1)
                values.append(subscore)
                board[i][j] = ''
                if subscore > best_score:
                    index = (i, j)
                    best_score = subscore
            else: values.append(-10)

    return index, values

def make_move(board, player, players, depth):
    if board.check_gameover():
        return -1
    move, values = find_move(board, player, players, depth)
    board[move[0]][move[1]] = player

    print("--------------\n| {0} | {4} | {8} | {12} |\n|-------------\n| {1} | {5} | {9} | {13} |\n|-------------\n| {2} | {6} | {10} | {14} |\n-------------\n| {3} | {7} | {11} | {15}".format(
        *values))

    return 0

def play_game():
    global n, m
    players = ['x', 'o']

    board = Board(verbose=True)

    while True:
        n, m = 0, 0
        print(board)

        print("Play move:")

        move = input()
        moves = list(map(int, re.findall('\d+', move)))
        board[moves[0]][moves[1]] = 'o'
        print(board)

        if make_move(board, 'x', players, 5) == -1:
            break

        print(board)

        if board.check_gameover() == True:
            break

if __name__ == "__main__":
    while True:
        play_game()
        print("To quit, press q. To play again, press any key.")
        if input() == 'q':
            break