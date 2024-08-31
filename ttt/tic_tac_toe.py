import random
import math

class TicTacToe:
    def __init__(self):
        self.board = tuple('.' for _ in range(9))
        self.current_player = 'X'

    def make_move(self, move):
        if self.board[move] == '.':
            new_board = list(self.board)
            new_board[move] = self.current_player
            self.board = tuple(new_board)
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            raise ValueError("Invalid move")

    def clone(self):
        new_game = TicTacToe()
        new_game.board = self.board
        new_game.current_player = self.current_player
        return new_game

    def __hash__(self):
        return hash((self.board, self.current_player))

    def __eq__(self, other):
        if isinstance(other, TicTacToe):
            return self.board == other.board and self.current_player == other.current_player
        return False

    def __str__(self):
        return '\n'.join(' '.join(self.board[i:i+3]) for i in (0, 3, 6))

    def is_terminal(self):
        return self.get_winner() is not None or '.' not in self.board

    def get_winner(self):
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != '.':
                return self.board[line[0]]
        if '.' not in self.board:
            return 'Tie'
        return None

    def get_possible_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == '.']

    def get_reward(self, player):
        winner = self.get_winner()
        print("getRewards")
        print(winner)
        print(player)
        if winner == 'O':
            return 5
        elif winner == 'Tie':
            return 4
        elif winner is None: 
            return 1
        else:
            return -10000

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_possible_moves())

    def select_child(self, exploration_weight):
        log_n_parent = math.log(self.visits)
        def uct(child):
            if child.visits == 0:
                return float('inf')
            return child.value / child.visits + exploration_weight * math.sqrt(log_n_parent / child.visits)
        return max(self.children, key=uct)

    def expand(self):
        move = random.choice([m for m in self.game.get_possible_moves() if not any(child.move == m for child in self.children)])
        child_game = self.game.clone()
        child_game.make_move(move)
        child_node = MCTSNode(child_game, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        print('Simulating')
        game = self.game.clone()
        visualize_board(game)
        while not game.is_terminal():
            move = random.choice(game.get_possible_moves())
            game.make_move(move)
            visualize_board(game)
        
        reward =  game.get_reward(self.game.current_player)
        print(f'Done simulating. Reward is : {reward}')
        return reward

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, exploration_weight=1):
        self.exploration_weight = exploration_weight

    def choose(self, root):

        def score(n):
            if n.visits == 0:
                return float("-inf")  # avoid unseen moves
            return n.value / n.visits  # average reward

        return max(root.children, key=score).move

    def select(self, node):
        while not node.game.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            node = node.select_child(self.exploration_weight)
        return node

    def rollout(self, root):
        for _ in range(100):  # number of iterations
            node = self.select(root)
            result = node.simulate()
            node.backpropagate(result)

    
    def get_stats(self, root):
        stats = {}
        for child in root.children:
            stats[child.move] = {
                'Q': child.value,
                'N': child.visits,
                'p': root.visits,
                'UCT': child.value / child.visits + self.exploration_weight * math.sqrt(math.log(root.visits) / child.visits) if child.visits > 0 else float('inf')
            }
        return stats


def visualize_board(game):
    board = game.board
    print("┌───┬───┬───┐")
    for i in range(0, 9, 3):
        print(f"│ {board[i]} │ {board[i+1]} │ {board[i+2]} │")
        if i < 6:
            print("├───┼───┼───┤")
    print("└───┴───┴───┘")

def visualize_mcts_stats(stats):
    print("\nMCTS Move Statistics:")
    print("┌─────┬──────┬──────┬──────┬──────┬──────┐")
    print("│ Move│   Q  │   N  │  Q/N │  p   │  UCT │")
    print("├─────┼──────┼──────┼──────┼──────┼──────┤")
    for move, data in stats.items():
        q_n_ratio = data['Q'] / data['N'] if data['N'] > 0 else 0
        print(f"│  {move}  │{data['Q']:5.2f} │{data['N']:5d} │{q_n_ratio:.4f}│{data['p']:5d} │{data['UCT']:5.4f} │")
    print("└─────┴──────┴──────┴──────┴──────┴──────┘")

def play_game(human_player='X'):
    game = TicTacToe()
    tree = MCTS()
    move_counter = 0

    while not game.is_terminal():
        print(f"\nMove {move_counter}")
        visualize_board(game)

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
            root = MCTSNode(game)
            tree.rollout(root)

            stats = tree.get_stats(root)
            visualize_mcts_stats(stats)

            move = tree.choose(root)
            game.make_move(move)
            print(f"AI chose move: {move}")

        move_counter += 1

    print("\nFinal board:")
    visualize_board(game)
    winner = game.get_winner()
    if winner == 'Tie':
        print("It's a tie!")
    else:
        print(f"Player {winner} wins!")

if __name__ == "__main__":
    play_game()