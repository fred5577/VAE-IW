import gym
import cv2
from collections import deque

class ResizeImage(gym.ObservationWrapper):
    def __init__(self, env, new_size):
        super(ResizeImage, self).__init__(env)
        self.resize_fn = lambda obs: cv2.resize(obs, dsize=new_size, interpolation=cv2.INTER_LINEAR)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_size)

    def observation(self, observation):
        return self.resize_fn(observation)


class FrameBuffer(gym.Wrapper):
    def __init__(self, env, buffer_size):
        assert (buffer_size > 0)
        super(FrameBuffer, self).__init__(env)
        self.buffer_size = buffer_size
        self.observations = deque(maxlen=buffer_size)
        shape = [self.buffer_size] + list(self.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)

    def step(self, a):
        observation, reward, done, info = self.env.step(a)
        self.observations.append(observation)
        return self.observation(), reward, done, info

    def reset(self):
        initial_frame = self.env.reset()
        for _ in range(self.buffer_size):
            self.observations.append(initial_frame)
        return self.observation()

    def observation(self):
        # Return a list instead of a numpy array to reduce space in memory when storing the same frame more than once
        return list(self.observations)


def is_atari_env(env):
    import gym.envs.atari
    return isinstance(env.unwrapped, gym.envs.atari.AtariEnv)


def wrap_atari_env(env, frameskip):
    # To get grayscale images, instead of wrapping the env, we modify the _get_obs function
    # this way, ale.getScreenGrayscale function is called instead of ale.getScreenRGB2
    # The RGB image will still show when rendering.
    screen_width, screen_height = env.unwrapped.ale.getScreenDims()
    env.unwrapped._get_obs = lambda : env.unwrapped.ale.getScreenGrayscale().reshape((screen_height, screen_width))
    env.unwrapped.frameskip = frameskip
    env = ResizeImage(env, new_size=(128, 128))
    # env = FrameBuffer(env, buffer_size=4)
    return env