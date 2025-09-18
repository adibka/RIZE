# Keep old Gym for your MuJoCo -v2 stack
import gym as gym_old
from gym.wrappers import TimeLimit  # if you use it elsewhere

# Bring in Gymnasium under a different name to avoid collisions
import gymnasium as gymn
import gymnasium_robotics as gymn_robotics

from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv, GymnasiumOldAPIAdapter

# Register Gymnasium-Robotics once
gymn.register_envs(gymn_robotics)

def _make_adroit_env(env_id: str):
    """Create Adroit envs from Gymnasium, then adapt to old-Gym API for RLKit."""
    base = gymn.make(env_id)                       # Gymnasium env
    env = GymnasiumOldAPIAdapter(base, env_id)     # old-Gym API surface
    return env

def make_env(name):
    # Route Adroit to Gymnasium; everything else stays on old Gym
    if name.startswith("AdroitHand"):
        env = _make_adroit_env(name)
    else:
        env = gym_old.make(name)

    env = CustomInfoEnv(env)
    env = NormalizedBoxEnv(env)
    return env



# import gym
# from gym.wrappers import TimeLimit

# from rlkit.envs.wrappers import CustomInfoEnv, NormalizedBoxEnv

# def make_env(name):
#     env = gym.make(name)
#     # Remove TimeLimit Wrapper
#     # if isinstance(env, TimeLimit):
#     #     env = env.unwrapped
#     env = CustomInfoEnv(env)
#     env = NormalizedBoxEnv(env)
#     return env