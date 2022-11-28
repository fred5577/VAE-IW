import argparse

parser = argparse.ArgumentParser(description='Planner for ROlloutIW')
parser.add_argument('--IW-method', default="rolloutIW", metavar='N',
                    help='Chose either IW or rolloutIW')
parser.add_argument('--features', default="bprost", metavar='N',
                    help='input to choose what methods to get features from')
parser.add_argument('--save-images', type=bool, default=False, metavar='N',
                    help='save images')
parser.add_argument('--number-of-saved-images', type=int, default=0, metavar='N',
                    help='The number of images to save before stopping execution.')
parser.add_argument('--env', default="SpaceInvaders-v4", metavar='N',
                    help='what enviroment to run on')
parser.add_argument('--frame-skip', type=int, default=15, metavar='N',
                    help='number of frameskips')           
parser.add_argument('--discount-factor', type=float, default=0.99, metavar='N',
                    help='Discount factor when picking actions')
parser.add_argument('--cache-subtree', type=bool, default=True, metavar='N',
                    help='Cache substree')
parser.add_argument('--max-nodes-generated', type=int, default=15000, metavar='N',
                    help='Number of nodes generated before the execution stops')
parser.add_argument('--test-round-of-model', type=bool, default=False, metavar='N',
                    help='Running the algorithm as a test round')
parser.add_argument('--model-name', default="../Results/SpaceInvaders/models/model_annealing_BCE_temp_1.pt", metavar='N',
                    help='The name of the model to use for features.')
parser.add_argument('--time-budget', type=float, default=0.5, metavar='N',
                    help='The name of the model to use for features.')
parser.add_argument('--zdim', type=int, default=1000, metavar='N',
                    help='Latent space dimension for the network used')
parser.add_argument('--use-negatives', type=bool, default=False, metavar='N',
                    help='Check if a feature is both true or false.')
parser.add_argument('--image-size-planning', type=int, default=84, metavar='N',
                    help='Check if a feature is both true or false.')
parser.add_argument('--xydim', type=int, default=4, metavar='N',
                    help='The x and x dim output of the model.')
parser.add_argument('--width', type=int, default=1, metavar='N',
                    help='Desired width when solving the problem. (Currently only 1 and 2 available)')
parser.add_argument('--rounds-to-run', type=int, default=5, metavar='N',
                    help='The number of rounds to run before ending.')
parser.add_argument('--risk-averse', type=bool, default=False, help='Run with risk averse rewards')
args = parser.parse_args()



def save_dataset_of_images(tree_actor, image_number, path_to_images, number_of_saved_images):
    nodes = random.sample(tree_actor.tree.nodes,number_of_saved_images)
    nodes.append(tree_actor.tree.root)
    for i in range(len(nodes)):
        path = path_to_images + "/image{0}.png".format(image_number + i)
        tree_actor.render(nodes[i].data["obs"], False, True, path, size=(160,210))
    return number_of_saved_images + 1

if __name__ == "__main__":
    import gym, gym.wrappers
    import numpy as np
    import os
    import sys
    import screen
    import gc
    import glob
    import time
    import random
    #import gridenvs.examples
    from sample import softmax_Q, sample_cdf
    from rolloutIW import RolloutIW
    from IW import IW
    from tree import TreeActor
    from atari_wrapper import wrap_atari_env
    from utils import env_has_wrapper, remove_env_wrapper
    
    print("Env: ", args.env, "Zdim: " , args.zdim, "Model name: ", args.model_name, "Features", args.features, " Time Budget: ", args.time_budget)


    # Check if pixel is background
    # HYPERPARAMETERS
    seed = 1234
    env_id = args.env
    discount_factor = args.discount_factor
    cache_subtree = args.cache_subtree
    max_nodes_generated = args.max_nodes_generated
    frame_skip = args.frame_skip
    save_images = args.save_images
    test_run = args.test_round_of_model
    time_budget = args.time_budget
    collect_pictures = 15000
    width = args.width
    risk_averse = args.risk_averse
    dirpath = os.getcwd()
    foldername = os.path.basename(dirpath)
    dirpath = dirpath.replace("/{0}".format(foldername), "")
    save_rewards = []
    # Random seed
    np.random.seed(seed)
    screenFeatures = screen.Screen(args.features, args.model_name, args.zdim, args.xydim, datasetsize=args.image_size_planning, use_neg=args.use_negatives)

    # Setting the path for where to save image
    if save_images:
        data_path = "{0}/Pictures-v1/{1}".format(dirpath,env_id)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_path = os.path.join(data_path)
        image_paths = glob.glob(data_path + '/*.png')
        step_count = len(image_paths)
    else:
        step_count = 0

    # Ready enviroment
    env = gym.make('ALE/'+env_id)
    # Creates the rollouts
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)
    env.reset()
    
    tree_actor = TreeActor(env, screenFeatures.GetFeatures)
    if args.IW_method is "rolloutIW":
        if args.features is "bprost":
            roll = RolloutIW(env.action_space.n, feature_size=20598848, width=width)
        else:
            if args.features == "model":
                roll = RolloutIW(env.action_space.n, feature_size=args.xydim**2 * args.zdim, width=width)
            elif args.features == "model_gaus":
                roll = RolloutIW(env.action_space.n, feature_size=args.xydim**2 * args.zdim * 3, width=width)
    else:
        roll = IW(env.action_space.n)

    tree = tree_actor.reset()
    
    number_of_rounds_to_run = args.rounds_to_run
    rounds = 0
    total_reward = 0
    round_reward = 0
    round_step = 0
    nodes_in_tree_average = 0
    pictures_collected = 0
    episode_done = False
    actions_applied = []
    while not env.unwrapped.ale.game_over():
        start_time = time.time()
        nodes_before_planning = len(tree_actor.tree)
        rollout_mean = roll.search(tree=tree,
                    successor_f=tree_actor.getSuccessor,
                    stop_time_budget=lambda current_time: current_time-start_time > time_budget)
        nodes_in_tree_average += len(tree)
        print("Nodes in tree before seleting action: ", len(tree), " Max Depth of tree: ", tree.max_depth, " Rollout Mean: ", np.mean(rollout_mean), " Number of rollouts: ", len(rollout_mean))
        
        # Save images 
        if save_images:
            pictures_collected = pictures_collected + save_dataset_of_images(tree_actor, pictures_collected, path_to_images="{0}/Pictures-v1/{1}".format(dirpath,env_id), number_of_saved_images=4)
            print("Pictures collected: ", pictures_collected)
            
        p = softmax_Q(tree, env.action_space.n, discount_factor, tree.root.data["ale.lives"], risk_averse)
        a = sample_cdf(p.cumsum())
        actions_applied.append(a)
        prev, curr = tree_actor.step(a, render=False, save_images=False, path_to_save_image="{0}/Pictures-v1/{1}/image{2}.png".format(dirpath,env_id,step_count))
     
        step_count += 1
        round_step += 1
        episode_done = curr["done"]
        total_reward += curr["r"]
        round_reward += curr["r"]
        print("Action: ", curr["a"], "Reward: ", curr["r"], ", Total reward: " , total_reward,
              ", Simulator steps: ", tree_actor.totalNodesGenerated, ", Planning steps:", step_count, ", Tree depth", tree.max_depth, 
               ", number of nodes in tree" , len(tree) ,"\n"
               ", size of feature vector: " , len(curr["features"]), ", Lives left: ", env.unwrapped.ale.lives(), 
               ", numbered of explored branches:", sum(p > 0), "\n" 
               ", round reward: ", round_reward, " round number: ", rounds, " Round steps: ", round_step, " Number of used features: ", roll.getUsedNoveltyFeatures())
        
        if number_of_rounds_to_run and (episode_done or env.unwrapped.ale.game_over() or round_step > max_nodes_generated):
            tree = tree_actor.reset()
            round_reward = 0
            round_step = 0
            rounds = rounds + 1
            # If save images is true we want to continue collecting until we have 15000 images
            if rounds >= number_of_rounds_to_run and not save_images:
                break
       
        if pictures_collected >= 15000:
            break
    
    print("Average over" , number_of_rounds_to_run, " is " , total_reward / number_of_rounds_to_run, "Average number of generated nodes is: ", nodes_in_tree_average/step_count)

    print("Game ended, Total reward" , total_reward , ", Simulator steps in total:" , tree_actor.totalNodesGenerated, ", Planning steps in total:" , step_count)
    # print("Why game ended, Gamer over:", 
    #     env.unwrapped.ale.game_over(), ", Max nodes generated: ", tree_actor.totalNodesGenerated < max_nodes_generated)
    print("Actions applied: " , actions_applied)
