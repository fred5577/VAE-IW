from collections import defaultdict
import numpy as np
import sys
import random

class Node:
    # Data {"prev_state" : features , "current_state" : features}
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.children = []
    
    def check_root(self):
        return self.parent is None

    def size(self):
        return np.sum([c.size() for c in self.children]) + 1

    def is_leaf(self):
        return len(self.children) == 0

    def add_data(self, data):
        return Node(data, parent=self)

    def create_root(self):
        if not self.check_root():
            self.parent.children.remove(self)
            self.parent = None
            temp_depth = self.depth
            for node in self.BFS():
                node.depth -= temp_depth 

    def BFS(self):
        node_self = [self]
        while len(node_self) > 0:
            children = []
            for node in node_self:
                yield node
                children.extend(node.children)
            node_self = children
    
    # def __str__(self):
    #     return "Action {0}, Reward {1}, Done {2}".format(self.data["a"], self.data["r"], self.data["done"])

class Tree:
    def __init__(self, root_data):
        self.createRoot(Node(root_data))

    def __len__(self):
        return len(self.nodes)

    def createRoot(self, node):
        node.create_root()
        self.root = node
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)
        for node in self.root.BFS():
            self._add_node(node)

    def _add_node(self, node):
        self.depth[node.depth].append(node)
        self.nodes.append(node)
        if node.depth > self.max_depth:
            self.max_depth = node.depth

    def add_node(self, parent, data):
        child = parent.add_data(data)
        self._add_node(child)
        return child

    def iter_BFS(self, include_root =True, include_leaves=True):
        if include_root:
            yield self.root
        for d in range(1, self.max_depth):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node

    def iter_BFS_reverse(self,include_root =False, include_leaves=True):
        for d in range(self.max_depth, 0, -1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node
        if include_root:
            yield self.root
        


class TreeActor:
    def __init__(self, env, getfeatures):
        self.env = env
        self.tree = None
        self.getfeatures = getfeatures
        self.totalNodesGenerated = 0
        self.totalSimulatorCallsThisRollout = 0
        self.done = True


    def getSuccessor(self, node, action):
        self.totalSimulatorCallsThisRollout += 1
        
        if self.last_node is not node:
            self.env.unwrapped.restore_state(node.data["s"])

        next_obs, reward, end_of_episode, info = self.env.step(action)
        node_for_data = {"a" : action, "r" : reward, "done" : end_of_episode, "obs" : next_obs}
        node_for_data.update(info)
        child = self.tree.add_node(node, node_for_data)
        self._update_state(node,child)
        self.totalNodesGenerated += 1
        return child

    def step(self, action, render=False, save_images=False, path_to_save_image=""):
        next_node = self._get_next_node(self.tree, action)
        root_data = self.tree.root.data

        self.tree.createRoot(next_node)
        self.done = next_node.data["done"]
        self._obs = root_data["obs"]
        self.totalSimulatorCallsThisRollout = 0
        if render or save_images: self.render(render=render, obs=self._obs, save_images=save_images, path_to_save_image=path_to_save_image)
        return root_data, next_node.data

    def reset(self):
        obs = self.env.reset()
        self.tree = Tree({"obs": obs, "done": False})
        self._update_state(None,self.tree.root)
        self.done = False
        return self.tree
    
    def _update_state(self, node, child):
        child.data["s"] = self.env.unwrapped.clone_state()
        child.data["ale.lives"] =self.env.unwrapped.ale.lives()
        self.getfeatures(self.env, child, node)
        self.last_node = child

    def _get_next_node(self, tree, action):
        next_node = None
        for child in tree.root.children:
            if action == child.data["a"]:
                next_node = child
                break
        assert next_node is not None, "Something wrong with the lookahead tree"

        return next_node

    def render(self, obs, render, save_images, path_to_save_image, size=(160,210)):
        import cv2
        img = obs[-1] if type(obs) is list else obs
        if size: img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #img = np.expand_dims(img, -1)
        if render:
            try:    
                self.viewer.imshow(img)
            except AttributeError:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        if save_images:
            #print(path_to_save_image)
            cv2.imwrite(path_to_save_image, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #return self.viewer.isopen
    
    def __del__(self):
        try:
            self.viewer.close()
        except AttributeError:
            pass


