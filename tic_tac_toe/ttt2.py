import random 
import math
import time 
from collections import defaultdict

class TicTacToe: 
    def __init__(self) -> None:
        self.board = ['.' for _ in range(9)]
        self.current_player = 'X'

    def get_possible_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == '.']

    def make_move(self, move): 
        if self.board[move] == '.': 
            self.board[move] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else: 
            raise ValueError("Invalid Move")

    def is_terminal(self): 
        return self.get_winner() is not None or '.' not in self.board

    def get_winner(self): 
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != '.':
                return self.board[line[0]]
        if '.' not in self.board:
            return 'Tie'
        return None

    def clone(self): 
        new_game = TicTacToe() 
        new_game.board = self.board.copy() 
        new_game.current_player = self.current_player
        return new_game

    def __str__(self) -> str:
        return '\n'.join(' '.join(self.board[i:i+3]) for i in (0, 3, 6))

    def __hash__(self):
        return hash(tuple(self.board) + (self.current_player,))

    def __eq__(self, other):
        return self.board == other.board and self.current_player == other.current_player

class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return random.choice(node.get_possible_moves())

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = set(node.get_possible_moves()) - self.children[node].keys()
            if unexplored:
                n = node.clone()
                n.make_move(unexplored.pop())
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = {}
        for move in node.get_possible_moves():
            child = node.clone()
            child.make_move(move)
            self.children[node][move] = child

    def _simulate(self, node):
        invert_reward = node.current_player == 'O'
        while True:
            if node.is_terminal():
                reward = self._get_reward(node)
                return 1 - reward if invert_reward else reward
            node = node.clone()
            node.make_move(random.choice(node.get_possible_moves()))
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        log_N_vertex = math.log(self.N[node])

        def uct(move):
            child_node = self.children[node][move]
            if self.N[child_node] == 0:
                return float('inf')
            return (self.Q[child_node] / self.N[child_node] +
                    self.exploration_weight * math.sqrt(log_N_vertex / self.N[child_node]))

        return self.children[node][max(self.children[node], key=uct)]

    def _get_reward(self, node):
        winner = node.get_winner()
        if winner == 'Tie':
            return 0.5
        return 1 if winner == 'X' else 0

def play_game(human_player='X', time_limit=10):
    game = TicTacToe()
    tree = MCTS()
    while not game.is_terminal():
        print(f"\nCurrent board:\n{game}")
        if game.current_player == human_player:
            while True:
                try:
                    move = int(input("Enter your move (0-8): "))
                    if move not in game.get_possible_moves():
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid move. Try again.")
            game.make_move(move)
        else:
            print("AI is thinking...")
            start_time = time.time()
            while time.time() - start_time < time_limit:
                tree.do_rollout(game)
            move = tree.choose(game)
            game.make_move(move)
            print(f"AI chose move: {move}")

    print(f"\nFinal board:\n{game}")
    winner = game.get_winner()
    if winner == 'Tie':
        print("It's a tie!")
    else:
        print(f"Player {winner} wins!")

if __name__ == "__main__":
    play_game()