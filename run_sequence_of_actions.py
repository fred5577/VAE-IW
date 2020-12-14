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
    import pandas as pd
    import csv
    #import gridenvs.examples

    from atari_wrapper import wrap_atari_env
    from utils import env_has_wrapper, remove_env_wrapper
    #env_id = "Robotank"
    actions = []
    env_id = "TimePilot"
    # number_actions = []

    with open("data/actions_applied.csv", 'r') as file:
        reader = csv.reader(file)
        skip_first = True
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            if row[0] == env_id:
                results = list(map(lambda x: int(x) if x != "nan" else int(0), row[1:]))
                actions.extend(results)
    print(actions)
    count = 0
    frame_skip = 15
    # Ready enviroment

    env = gym.make(env_id + "-v4")
    env = gym.wrappers.Monitor(env, '/work1/s153430/Master/videos', force = True)
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)
    env.reset()

    for i in range(len(actions)):
        env.step(actions[i])
    env.close()
        
        
