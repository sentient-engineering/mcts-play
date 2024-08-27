import random
import math
import time
from collections import defaultdict

class LinearArrayGame:
    def __init__(self, start=1, end=4):
        self.position = start
        self.end = end
        self.max_move = 4  # Maximum number of steps allowed in one move

    def get_possible_moves(self):
        return [1,2,3]

    def make_move(self, move):
        if move in self.get_possible_moves():
            self.position += move
        else:
            raise ValueError("Invalid Move")

    def is_terminal(self):
        return self.position >= self.end

    def get_result(self):
        return 1 if self.position == self.end else 0

    def clone(self):
        new_game = LinearArrayGame(self.position, self.end)
        return new_game

    def __str__(self):
        return f"Current position: {self.position}, End: {self.end}"

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return isinstance(other, LinearArrayGame) and self.position == other.position

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

        def score(move):
            child_node = self.children[node][move]
            if self.N[child_node] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[child_node] / self.N[child_node]  # average reward

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
                move = unexplored.pop()
                child = node.clone()
                child.make_move(move)
                self.children[node][move] = child
                path.append(child)
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
        while not node.is_terminal():
            node = node.clone()
            node.make_move(random.choice(node.get_possible_moves()))
        return node.get_result()

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        log_N_vertex = math.log(self.N[node])

        def uct(move):
            child_node = self.children[node][move]
            if self.N[child_node] == 0:
                return float('inf')
            return (self.Q[child_node] / self.N[child_node] +
                    self.exploration_weight * math.sqrt(log_N_vertex / self.N[child_node]))

        return self.children[node][max(self.children[node].keys(), key=uct)]

def find_optimal_path(start=1, end=20, time_limit=5):
    game = LinearArrayGame(start, end)
    tree = MCTS()
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        tree.do_rollout(game)
    
    path = []
    current_game = game
    while not current_game.is_terminal():
        best_move = tree.choose(current_game)
        path.append(best_move)
        current_game.make_move(best_move)
    
    return path

def main():
    start = 1
    end = 4
    optimal_path = find_optimal_path(start, end)
    
    print(f"Optimal path from {start} to {end}:")
    current_position = start
    for move in optimal_path:
        print(f"Move {move} steps from {current_position} to {current_position + move}")
        current_position += move
    print(f"Reached the end at position {current_position}")
    print(f"Total moves: {len(optimal_path)}")

if __name__ == "__main__":
    main()