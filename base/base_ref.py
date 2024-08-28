from collections import defaultdict
from abc import ABC, abstractmethod
import math

class Node(ABC):
    @abstractmethod
    def find_children(self): 
        return set() 
    
    @abstractmethod
    def find_random_child(self):
        return None
    
    @abstractmethod
    def is_terminal(self): 
        return True
    
    @abstractmethod
    def reward(self): 
        return 0 
    
    @abstractmethod
    def __hash__(self) -> int:
        return 123456789
    
    @abstractmethod
    def __eq__(self, value: object) -> bool:
        return True
    


class MCTS: 
    def __init__(self, exploration_weight=1) -> None:
        self.reward = defaultdict(int)
        self.visits = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight

    def choose(self, node: Node): 
        "Choose the best successor of node (Choose a move in the game)"
        if node.is_terminal(): 
            raise RuntimeError(f"choose called on terminal node {node}")
        
        if node not in self.children: 
            return node.find_random_child()
        
        def score(node: Node): 
            if self.visits[node] == 0:
                return float("-inf")
            return self.reward[node]/self.visits[node]
        
        return max(self.children[node], key=score)
    
    def _uct_select(self, node: Node): 
        "Select a child of node, balancing exploration and exploitation"
        # All the children of the node should already be expanded
        assert all (n in self.children for n in self.children[node])

        log_n_vertext = math.log(self.visits[node])

        def uct(node: Node): 
            "Upper confidence bound for trees"
            return self.reward[node]/ self.visits[node] + self.exploration_weight * math.sqrt(log_n_vertext / self.visits[node])
        
        return max(self.children[node], key = uct)
    

    def _select(self, node: Node): 
        "Find an unexplored descendent of the node"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal 
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored: 
                n = unexplored.pop()
                path.append(n)
                return path 
            node = self._uct_select(node) # descend a layer deeper 

    def _expand(self, node: Node): 
        "Update the `childen` dict with the children of `node`"
        if node in self.children: 
            return # already expanded 
        self.children[node] = node.find_children()

    def _simulate(self, node: Node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True: 
            if node.is_terminal():
                reward = node.reward() 
                return 1 - reward if invert_reward else reward
            
            node = node.find_random_child() 
            invert_reward = not invert_reward

    def _backpropogate(self, path, reward): 
        "Send the reward back upto the ancestors of the leaf"
        for node in reversed(path): 
            self.visits[node] += 1 
            self.reward[node] += reward
            reward = 1 - reward # 1 for me is 0 for my enemy and vice versa

    def do_rollout(self, node: Node): 
        "Make the tree  one layer better. Train for one iteration"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropogate(path, reward)



