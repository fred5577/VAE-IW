import random
from collections import defaultdict
import numpy as np
from sample import sample_cdf, softmax_Q, softmax
import time

class IW:
    def __init__(self, branching_factor, ignore_tree_caching = False, ignore_terminal_nodes=False):
        #"Initializing a rollout"
        # Create novelty table
        self.branching_factor = branching_factor
        self.ignore_terminal_nodes = ignore_terminal_nodes
        self.ignore_tree_caching = ignore_tree_caching
        self.width = 1

    # Used for online planing (Generally the Atari domain is only planning since we have a window to plan in)
    def initialize(self, tree):
        for node in tree.nodes:
            assert "features" in node.data.keys(), "IW planners require the state to be factored into features"
            self.novelty_table.check_and_update_novelty_table(node.data["features"])
            #print(len(node.children))
            if len(node.children) is not self.branching_factor:
                temp = [(node, a) for a in range(len(node.children), self.branching_factor)]
                self.queue.extend(temp)
                

    def search(self, tree, successor_f, stop_simulator=lambda:False, stop_time_budget=lambda:False, time_budget=5):
        # tree : state tree
        # successor_f : generates a successor for a node
        # stop : the function that indicates the budget
        #"Write the while loop where we are performing rollouts till we have a terminal node or time runs out"
        self.novelty_table = NoveltyTable(self.ignore_tree_caching)

        self.queue = []
        self.initialize(tree)
        count= 0
        while not stop_simulator() and len(self.queue) != 0 and not stop_time_budget(time.time()):
            #"Get action from root node"
            (node, a) = self.queue.pop(0)

            node = successor_f(node,a)
            novel = self.novelty_table.check_and_update_novelty_table(node.data["features"])
            if novel:
                temp = [(node, a) for a in range(self.branching_factor)]
                self.queue.extend(temp)
                #print(len(self.queue))
   

class NoveltyTable:
    """
    This holds the novelty of all the atoms that have currently been set to true and at what depth, they are.
    """
    def __init__(self, ignore_tree_caching):
        self.table = defaultdict(lambda: False)
        # TODO : Ignore tree caching is currently not used.
        self.ignore_tree_caching = True

    def check_table(self, features):
        for feature in features:
            if not self.table[feature]:
                return True # cases 1 and 4 from the article (the rollout continues)
        return False # all atoms are either case 2 or 3, so the rollout stops.

    def check_and_update_novelty_table(self, features):
        is_novel = False
        for feature in features:
            if not self.table[feature]:
                is_novel = True
            self.table[feature] = True
        return is_novel




