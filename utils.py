def env_has_wrapper(env, wrapper_type):
    while env is not env.unwrapped:
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return False

def remove_env_wrapper(env, wrapper_type):
    if env is not env.unwrapped:
        if isinstance(env, wrapper_type):
            env = remove_env_wrapper(env.env, wrapper_type)
        else:
            env.env = remove_env_wrapper(env.env, wrapper_type)
    return env