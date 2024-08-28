from base.base import MCTS, Node
from random import choice

class LinearArrayBoard(Node):
    def __init__(self, position=1, end=50, turn=True):
        self.position = position
        self.end = end
        self.turn = turn
        self.terminal = self.position >= self.end

    def find_children(board):
        if board.terminal:
            return set()
        return {board.make_move(move) for move in [1, 2, 3]}

    def find_random_child(board):
        if board.terminal:
            return None
        return board.make_move(choice([1, 2, 3]))

    def is_terminal(board):
        return board.terminal

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        return 1 if board.position == board.end else 0

    def make_move(board, move):
        new_position = min(board.position + move, board.end)
        return LinearArrayBoard(new_position, board.end, not board.turn)

    def __hash__(self):
        return hash((self.position, self.turn))

    def __eq__(self, other):
        return (self.position, self.turn) == (other.position, other.turn)

def play_game():
    board = LinearArrayBoard()
    tree = MCTS()
    
    print(f"Starting position: {board.position}")
    while True:
        for _ in range(20000):  # Number of MCTS iterations
            tree.do_rollout(board)
        
        board = tree.choose(board)
        print(f"MCTS chose to move to position: {board.position}")
        
        if board.terminal:
            break
    
    print(f"Game over! Final position: {board.position}")
    print("Goal reached!" if board.position == board.end else "Goal not reached.")

if __name__ == "__main__":
    play_game()