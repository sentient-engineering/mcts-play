import math
import random

class LinearArrayGame:
    def __init__(self, start=1, end=20):
        self.position = start
        self.end = end
        self.max_move = 2  # Maximum number of steps allowed in one move

    def get_possible_moves(self):
        return [1, 2, 3]  # Can move either 1 or 2 steps

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

def mcts(root: Node, iterations):
    for iter in range(iterations):
        # Select will return a leaf node that can either be terminal or nor terminal.  
        node = select(root)
        # Rollout will play the game from obtained leaf node randonly till a terminal state is reached. 
        score = rollout(node.game)
        backpropagate(node, score)
    return best_child(root).move

# Select and Expand work together like this - 
# a) Select traverses the tree till a leaf node is reached 

# b) if the leaf node is non-terminal, expand is called to add children to this lead node (as its not a non-terminal leaf, its bound to have more children)
# d) Expand, after adding children to tree, randomly returns a child  for rollout ie. the game is played from this node till a terminal state is reached.  
# c) The new child is added for rollout ie we play game randomly from here till a terminal state
# Note: Select will typically return a non terminal leaf node. It only retuns a terminal leaf node, if the entire subtree from root till the node has been expanded. 
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
    return (node.score / node.visits) + math.sqrt(2 * math.log(node.parent.visits) / node.visits)

def best_child(node):
    return max(node.children, key=lambda n: n.visits)

def play_game():
    game = LinearArrayGame()
    # Start of Turn: At the beginning of each turn, we create a new root node representing the current game state.
    # Run MCTS: We run MCTS for a fixed number of iterations (1000 in this case) to determine the best move from this position.
    # Make Move: We make the best move found by MCTS.
   # Repeat: We then repeat this process for the new game state, until the game ends. This is specially relevant in multiplayer game as the state changes post the opponent makes their move and whole calculation need to be redone for the new state 
    while not game.is_terminal():
        print(f"Current position: {game.position}")
        root = Node(game)
        best_move = mcts(root, 100000)  # Perform 1000 MCTS iterations
        game.make_move(best_move)
        print(f"MCTS chose to move {best_move} step(s)")
    print(f"Game over! Final position: {game.position}")
    print("Goal reached!" if game.get_result() == 1 else "Goal not reached.")

if __name__ == "__main__":
    play_game()