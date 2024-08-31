import random
import math

class LinearArrayState:
    def __init__(self, position=1, goal=20):
        self.position = position
        self.goal = goal

    def find_children(self):
        children = []
        for move in [1, 2, 3]:
            new_position = self.position + move
            if new_position <= self.goal:
                children.append(LinearArrayState(new_position, self.goal))
        return children

    def find_random_child(self):
        valid_moves = [move for move in [1, 2, 3] if self.position + move <= self.goal]
        if not valid_moves:
            return None
        move = random.choice(valid_moves)
        return LinearArrayState(self.position + move, self.goal)

    def is_terminal(self):
        return self.position == self.goal

    def reward(self):
        return 1 if self.position == self.goal else 0

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"Position: {self.position}"

class MCTS:
    def __init__(self, exploration_weight=1.4):
        self.Q = {}
        self.N = {}
        self.children = {}
        self.exploration_weight = exploration_weight

    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

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
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = self._pick_best_move(node)
            invert_reward = not invert_reward

    def _pick_best_move(self, node):
        children = node.find_children()
        return max(children, key=lambda n: n.position)

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] = self.N.get(node, 0) + 1
            self.Q[node] = self.Q.get(node, 0) + reward
            reward = 1 - reward  # 1 for me is 0 for my opponent

    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

def play_game():
    state = LinearArrayState()
    mcts = MCTS()

    print("Starting Linear Array Game (1 to 20)")
    print(f"Initial {state}")

    while not state.is_terminal():
        for _ in range(100000):  # Increased number of MCTS iterations
            mcts.do_rollout(state)
        
        best_move = mcts.choose(state)
        move_distance = best_move.position - state.position
        state = best_move
        
        print(f"Best move: +{move_distance} -> {state}")

    print("Goal reached!")

if __name__ == "__main__":
    play_game()