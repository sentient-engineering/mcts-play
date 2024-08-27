import math
import random

class LinearArrayGame:
    def __init__(self, start=1, end=4):
        self.position = start
        self.end = end
        self.max_move = 2  # Maximum number of steps allowed in one move

    def get_possible_moves(self):
        return [1, 2]  # Can move either 1 or 2 steps

    def make_move(self, move):
        self.position += move
        return self.position

    def is_terminal(self):
        return self.position >= self.end

    def get_result(self):
        return 1 if self.position == self.end else 0

    def clone(self):
        new_game = LinearArrayGame(self.position, self.end)
        return new_game

class Node:
    def __init__(self, game, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

def mcts(root, iterations):
    for _ in range(iterations):
        node = select(root)
        score = rollout(node.game)
        backpropagate(node, score)
    return best_child(root).move

def select(node):
    while not node.game.is_terminal():
        if not node.children:
            return expand(node)
        node = uct_select(node)
    return node

def expand(node):
    moves = node.game.get_possible_moves()
    for move in moves:
        new_game = node.game.clone()
        new_game.make_move(move)
        new_node = Node(new_game, move, node)
        node.children.append(new_node)
    return random.choice(node.children)

def rollout(game):
    while not game.is_terminal():
        move = random.choice(game.get_possible_moves())
        game.make_move(move)
    return game.get_result()

def backpropagate(node, score):
    while node:
        node.visits += 1
        node.score += score
        node = node.parent

def uct_select(node):
    return max(node.children, key=lambda n: uct_value(n))

def uct_value(node):
    if node.visits == 0:
        return float('inf')
    return node.score / node.visits + math.sqrt(2 * math.log(node.parent.visits) / node.visits)

def best_child(node):
    return max(node.children, key=lambda n: n.visits)

def play_game():
    game = LinearArrayGame()
    while not game.is_terminal():
        print(f"Current position: {game.position}")
        root = Node(game)
        best_move = mcts(root, 1000)  # Perform 1000 MCTS iterations
        game.make_move(best_move)
        print(f"MCTS chose to move {best_move} step(s)")
    print(f"Game over! Final position: {game.position}")
    print("Goal reached!" if game.get_result() == 1 else "Goal not reached.")

if __name__ == "__main__":
    play_game()