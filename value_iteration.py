'''
This code is an implementation of vaue iteration in the Grid World.
'''
__author__ = 'Minsoo Kim'
__copyright__ = 'Copyright 2019, Simple Reinforcement Learning project'
__credits__ = ['Jaemin Seol']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Minsoo Kim'
__email__ = 'msk930512@{snu.ac.kr, gmail.com}'
__status__ = 'Development'

import numpy as np

grid_width = 4
grid_height = 4
discount = 1.
reward = -1.
action = [[-1,0], [1,0], [0, -1], [0, 1]]# up, down, left, right
action_string = ['U', 'D', 'L', 'R']
policy = np.full((grid_height*grid_width, len(action)), [0.25, 0.25, 0.25, 0.25])# initial policy
policy_evaluation_iteration = 1000

TOTAL_ITERATION = 10
GRID_RENDER = True


class GridWorldMDP():
    def __init__(self, grid_width, grid_height, immediate_reward, discount_factor):
        self.states = np.zeros((grid_height, grid_width))
        self.r = immediate_reward
        self.dis_f = discount_factor

class ValueIteration():
    def __init__(self, MDP, action, init_policy):
        self.action = action
        self.policy = init_policy
        self.MDP = MDP
    
    def value_iteration(self, iteration):
        for _ in range(iteration):
            value = np.zeros((grid_height, grid_width))
            for i in range(grid_height):
                for j in range(grid_width):
                    adjacent_values = np.zeros(4)
                    for idx, act in enumerate(action):
                        if (i == 0 and j == 0) or (i == grid_height-1 and j == grid_width-1):# Terminal State
                            value[i, j] = 0
                            continue
                        row = i + act[0] if (i + act[0] >= 0) and (i + act[0] < grid_height) else i
                        column = j + act[1] if (j + act[1] >= 0) and (j + act[1] < grid_width) else j
                        adjacent_values[idx] = reward + discount*self.MDP.states[row, column]
                    value[i, j] = round(np.amax(adjacent_values), 3)
            self.MDP.states = value
            if GRID_RENDER:
                print("@@@@@ {} Iteration ====================".format(_))
                print(self.MDP.states)
        return self.MDP.states


def main():                                           
    print("==================== Value Iteration ====================")
    grid_world = GridWorldMDP(grid_width, grid_height, reward, discount)
    agent = ValueIteration(grid_world, action, policy)
    agent.value_iteration(TOTAL_ITERATION)

if __name__=="__main__":
    main()
