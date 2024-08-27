import random 
import math
import time 

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
            raise ValueError("Invalid Error")

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

    def get_result(self, player): 
        winner = self.get_winner()
        if winner == player: 
            return 1 
        elif winner == 'Tie': 
            return 0.5
        else:
            return 0

    def clone(self): 
        new_game = TicTacToe() 
        new_game.board = self.board.copy() 
        new_game.current_player = self.current_player
        return new_game

    def __str__(self) -> str:
        return '\n'.join(' '.join(self.board[i:i+3]) for i in (0, 3, 6)) 

class Node: 
    def __init__(self, game: TicTacToe, move=None, parent=None) -> None:
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0 
        self.score = 0 

def expand(node: Node): 
    moves = node.game.get_possible_moves()
    for move in moves: 
        new_game = node.game.clone()
        new_game.make_move(move)
        new_node = Node(new_game, move, node)
        node.children.append(new_node)

    return random.choice(node.children)

def uct_select(node: Node): 
    return max(node.children, key=lambda node: uct_value(node))

def uct_value(node: Node): 
    if node.visits == 0: 
        return float('inf')
    return node.score / node.visits + math.sqrt(2 * math.log(node.parent.visits) / node.visits)

def select(node: Node): 
    while not node.game.is_terminal(): 
        if not node.children: 
            return expand(node)
        node = uct_select(node)

    return node 

def rollout(game: TicTacToe): 
    current_game = game.clone()
    while not current_game.is_terminal(): 
        move = random.choice(current_game.get_possible_moves())
        current_game.make_move(move)

    return current_game.get_result(game.current_player)

def backpropogate(node: Node, score):
    while node: 
        node.visits += 1 
        node.score += score
        node = node.parent
        score = 1 - score

def get_best_move(node: Node): 
    return max(node.children, key=lambda node : node.visits).move 

def mcts(root: Node, iterations, time_limit): 
    end_time = time.time() + time_limit
    for _ in range(iterations): 
        if time.time() > end_time: 
            break

        leaf = select(root)
        score = rollout(leaf.game)
        backpropogate(leaf, score)

    return get_best_move(root)

def play_game(human_player='X', time_limit=10): 
    game = TicTacToe()
    while not game.is_terminal(): 
        print(f"\nCurrent board:\n{game}")
        if game.current_player == human_player:
            while True:
                try:
                    move = int(input("Enter your move (0-8): "))
                    game.make_move(move)
                    break
                except ValueError:
                    print("Invalid move. Try again.")
        else: 
            print("AI is thinking...")
            root = Node(game)
            move = mcts(root, iterations=10000, time_limit=time_limit)
            game.make_move(move)
            print(f"AI chose move: {move}")

    print(f"\n Final board:\n{game}")
    winner = game.get_winner()
    if winner == 'Tie': 
        print("Its a tie!")
    else: 
        print(f"Player {winner} wins!")

if __name__ == "__main__": 
    play_game()