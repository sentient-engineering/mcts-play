from base.base_ref import MCTS, Node
from collections import namedtuple
from random import choice

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

# Inheriting from a namedtuple is conviniet because it makes the class immutable and predefines __init__, __repr__, __hash__ , __eq__ and others 

class TicTacToeBoard(_TTTB, Node): 
    def find_children(board):
        if board.terminal: 
            return set()
        return {
            board.make_move(i) for i, value in enumerate (board.tup) if value is None
        }
    
    def find_random_child(board):
        if board.terminal: 
            return None
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))
    
    def reward(board): 
        if not board.terminal: 
            raise RuntimeError(f"reward called on nonterminal board{board}")
        if board.winner is board.turn: 
            raise RuntimeError(f"reward called on unreacahble board {board}")
        if board.turn is (not board.winner): 
            return 0 
        if board.winner is None: 
            return 0.5 
        raise RuntimeError(f"Board has unknown winner type {board.winner}")
    
    def is_terminal(board):
        return board.terminal
    
    def make_move(board, index):
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )
    
def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None

def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal       


def play_game():
    tree = MCTS()
    board = new_ttt_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        board = board.make_move(index)
        print(board.to_pretty_string())
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollot(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break



def new_ttt_board(): 
    return TicTacToeBoard(tup=(None, ) * 9, turn = True, winner = None, terminal = False)


if __name__ == "__main__": 
    play_game()