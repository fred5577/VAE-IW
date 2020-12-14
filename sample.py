import numpy as np

def sample_cdf(cum_probs):
    s = cum_probs[-1]
    assert s > 0.99999 and s < 1.00001, "Probabilities do not sum to 1: %"%cum_probs #just to check our input looks like a probability distribution, not 100% sure though.
    while True:
        rand = np.float32(np.random.rand())
        if rand < s:
            break
    res = (cum_probs > rand)
    return res.argmax()

def compute_return(tree, discount_factor,current_lives, risk_averse):
    for node in tree.iter_BFS_reverse():
        if node.is_leaf():
            if node.data["ale.lives"] < current_lives:
                R = -10 * 50000 + node.data["r"] if risk_averse else node.data["r"]
            elif node.data["r"] < 0:
                R = node.data["r"] * 50000 if node.data["r"] else node.data["r"]
            else:
                R = node.data["r"]
        else:
            if node.data["ale.lives"] < current_lives:
                cost = -10 * 50000 if risk_averse else 0
                R = cost + node.data["r"] + discount_factor * np.max([child.data["R"] for child in node.children])
            elif node.data["r"] < 0:
                cost = node.data["r"] * 50000 if risk_averse else node.data["r"]
                R = cost + discount_factor * np.max([child.data["R"] for child in node.children])
            else:
                R = node.data["r"] + discount_factor * np.max([child.data["R"] for child in node.children])
        node.data["R"] = R

def softmax(x, temp=1, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    if temp == 0:
        res = (x == np.max(x, axis=-1))
        return res/np.sum(res, axis=-1)
    x = x/temp
    e_x = np.exp( (x - np.max(x, axis=axis, keepdims=True)) ) #subtracting the max makes it more numerically stable, see http://cs231n.github.io/linear-classify/#softmax and https://stackoverflow.com/a/38250088/4121803
    return e_x / e_x.sum(axis=axis, keepdims=True)

def softmax_Q(tree, branching, discount_factor, current_lives, risk_averse):
    temp=0
    compute_return(tree, discount_factor, current_lives, risk_averse)
    Q = np.empty(branching, dtype=np.float32)
    Q.fill(-np.inf)
    for child in tree.root.children:
        Q[child.data["a"]] = child.data["R"]
    return softmax(Q, temp=0)