import random
from collections import defaultdict
import numpy as np
from sample import sample_cdf, softmax_Q, softmax
import time

class RolloutIW:
    def __init__(self, branching_factor, feature_size, width=1, ignore_tree_caching = False, ignore_terminal_nodes=False):
        self.branching_factor = branching_factor
        self.ignore_terminal_nodes = ignore_terminal_nodes
        self.ignore_tree_caching = ignore_tree_caching
        self.width = width
        self.feature_size = feature_size
        self.rollouts = []
    
    def getUsedNoveltyFeatures(self):
        return len(self.novelty_table.table[self.novelty_table.table < np.inf])

    # Used for online planing (Generally the Atari domain is only planning since we have a window to plan in)
    def initialize(self, tree):

        for node in tree.iter_BFS():
            assert "features" in node.data.keys(), "IW planners require the state to be factored into features"
            node.solved = False

            if node.data["done"]:
                self.solve_and_propagate(node)

    def search(self, tree, successor_f, stop_simulator=lambda:False, stop_time_budget=lambda:False):
        # Create novelty table
        self.novelty_table = NoveltyTable(self.ignore_tree_caching,feature_size = self.feature_size, width=self.width)

        policy = lambda n, bf: np.full(bf, 1 / bf) # Uniformaly distrubtion of actions

        self.initialize(tree)
        while not stop_simulator() and not tree.root.solved and not stop_time_budget(time.time()):
            #"Get action from root node"
            self.rollout_length = 0
            a,node = self.select(tree.root, policy(tree.root, self.branching_factor))
            self.rollout_length = self.rollout_length + 1
            #Perform rollout from root node with that action
            if a is not None:
                self.rollout(node, a, successor_f, stop_simulator,stop_time_budget, policy)
        mean = self.rollouts
        self.rollouts = []
        return mean
        
    
    def rollout(self, node, action, successor_f, stop_simulator,stop_time_budget, policy):
        #"Some stop condition"
        while not stop_simulator() and not stop_time_budget(time.time()):
            # Get a successor for node and action
            node = successor_f(node, action)
            node.solved = False

            if node.data["done"]:
                # Rollout done
                # Update novelty table
                self.novelty_table.check_and_update_novelty_table(node.data["features"], node.depth, new_node = True)
                # Set node to solved and propogate
                self.solve_and_propagate(node)
                self.rollouts.append(self.rollout_length)
                return
            else:
                # Update novelty table
                novel = self.novelty_table.check_and_update_novelty_table(node.data["features"], node.depth, new_node= True)
                # Check if width is less then novelty
                if novel > self.width:
                    # Solved node and propogate
                    self.solve_and_propagate(node)
                    self.rollouts.append(self.rollout_length)
                    return
            # Other wise pick an new random action
            action, child = self.select_action_policy(node, policy(node, self.branching_factor))
            self.rollout_length = self.rollout_length + 1
            #assert action is not None and child is None, "Action: %s, child: %s"%(str(action), str(child))
        return
    

    def select(self, node, policy):
        # TODO : consider the possiblity of their being no possible actions (leaf node)
        assert not node.data["done"]
        while True:
            novel = self.novelty_table.check_and_update_novelty_table(node.data["features"], node.depth, new_node = False)
            if novel > self.width:
                self.solve_and_propagate(node)
                return None, None # Prune node
            
            a, child = self.select_action_policy(node, policy)
            if a is None:
                return None, None # All possible actions have been solved
            else:
                if child is None:
                    return a, node
                else:
                    node = child


    def select_action_policy(self, node, policy):
        if node.is_leaf():
            a = random.randrange(self.branching_factor)
            return a, None
        avail_actions = (policy > 0)
        node_children = [None]*self.branching_factor
        for child in node.children:
            node_children[child.data["a"]] = child
            if child.solved:
                avail_actions[child.data["a"]] = False

        probability_distribution = (policy*avail_actions)
        sum_pd = probability_distribution.sum()
        if sum_pd <= 0:
            self.solve_and_propagate(node)
            return None, None
        probability_distribution = probability_distribution/sum_pd
        a = sample_cdf(probability_distribution.cumsum())
        return_child = node_children[a]

        if return_child:
            assert not return_child.solved

        return a, return_child

    def check_parent_solved(self, node):
        if len(node.children) == self.branching_factor and all(child.solved for child in node.children):
            return True
        return False

    def solve_and_propagate(self, node):
        node.solved = True
        while not node.check_root():
            node = node.parent
            if self.check_parent_solved(node):
                node.solved = True
            else:
                break
            

class NoveltyTable:
    def __init__(self, ignore_tree_caching, feature_size, width=1):
        self.table = np.full(feature_size**width, np.inf)
        # TODO : Ignore tree caching is currently not used.
        self.ignore_tree_caching = True
        self.width = width
        self.feature_size = feature_size
        self.counter =0
        

    def check_table(self, features, depth, new_node):
        for feature in features:
            if depth < self.table[feature] or (not new_node and depth == self.table[feature]):
                return 1 # cases 1 and 4 from the article (the rollout continues)
        return np.inf # all atoms are either case 2 or 3, so the rollout stops.

    def check_and_update_novelty_table(self, features, depth, new_node):
        is_novel = False
        
        def condition(x): 
            return depth < x  
        def condition1(x): 
            return depth == x 

        case1 = features[np.argwhere(condition(self.table[features])).flatten()]
        case2 = not new_node and (features[np.argwhere(condition1(self.table[features])).flatten()]).any()

        if len(case1) > 0:
            is_novel = True
            if new_node:
                self.table[case1] = depth

        if case2 and len(case1) == 0:
            is_novel = True

        return 1 if is_novel else np.inf
        


