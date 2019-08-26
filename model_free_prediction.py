'''
This code is an implementation of model-free prediction in the FrozenLake-v0.
1. Monte-Carlo Prediction
2. Temporal-Difference (TD) Prediction
3. n-step TD
4. TD (lambda)
'''
__author__ = 'Minsoo Kim'
__copyright__ = 'Copyright 2019, Simple Reinforcement Learning project'
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Minsoo Kim'
__email__ = 'msk930512@{snu.ac.kr, gmail.com}'
__status__ = 'Development'

import gym
import numpy as np
MAX_ITERATIONS = 5000
RENDER = False
from gym.envs.registration import register
register(
   id='GoldMine-v0',
   entry_point='gym.envs.toy_text:FrozenLakeEnv',
   kwargs={'map_name' : '4x4', 'is_slippery': False},
   max_episode_steps=100,
   reward_threshold=0.78, # optimum = .8196
)
env = gym.make("GoldMine-v0")
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
MCvalue_prediction = np.zeros(16)
TDvalue_prediction = np.zeros(16)
step_size = 0.01
discount_factor = 1
number_of_visit = np.zeros(16)
for i in range(MAX_ITERATIONS):
   env.reset()
   env.render()
   state = 0
   returnGt = 0
   trajectory = []
   done = False
   while not done:
       random_action = env.action_space.sample()
       new_state, reward, done, info = env.step(random_action)
       TD_target = reward + discount_factor*TDvalue_prediction[new_state]
       TDvalue_prediction[state] += step_size*(TD_target - TDvalue_prediction[state])
       trajectory.append(new_state)
       state = new_state
       env.render()
   # estimating the value function by the episode.
   print(trajectory)
   unique, counts = np.unique(trajectory, return_counts =  True)
   n = dict(zip(unique, counts))
   returnGt = reward
   for unique, count in n.items():
       number_of_visit[unique] += count
       MCvalue_prediction[unique] += (returnGt - MCvalue_prediction[unique])/number_of_visit[unique]
   env.render()
   print(number_of_visit)
env.reset()
env.render()
print("@@@@@ MC ===============")
print(np.reshape(MCvalue_prediction,(4,4)))
print("@@@@@ TD ===============")
print(np.reshape(TDvalue_prediction,(4,4)))