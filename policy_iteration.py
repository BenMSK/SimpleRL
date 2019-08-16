'''
This code is an implementation of policy iteration in the Grid World.
'''
__author__ = 'Minsoo Kim'
__copyright__ = 'Copyright 2019, Simple Reinforcement Learning project'
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

TOTAL_ITERATION = 10
GRID_RENDER = True

class GridWorldMDP():
    def __init__(self, grid_width, grid_height, immediate_reward, discount_factor):
        self.states = np.zeros((grid_height, grid_width))
        self.r = immediate_reward
        self.dis_f = discount_factor

class PolicyIteration():
    def __init__(self, MDP, action, init_policy):
        self.action = action
        self.policy = init_policy
        self.MDP = MDP
        self.policy_evaluation_iteration = 1000
        
    def policy_evaluation(self):
        for _ in range(self.policy_evaluation_iteration):
            value = np.zeros((grid_height, grid_width))
            count = 0
            for i in range(grid_height):
                for j in range(grid_width):
                    for idx, act in enumerate(action):
                        if (i == 0 and j == 0) or (i == grid_height-1 and j == grid_width-1):# Terminal State
                            value[i, j] = 0
                            continue
                        row = i + act[0] if (i + act[0] >= 0) and (i + act[0] < grid_height) else i
                        column = j + act[1] if (j + act[1] >= 0) and (j + act[1] < grid_width) else j
                        value[i, j] += round(self.policy[count, idx]*(reward + discount*self.MDP.states[row, column]), 3)
                    count += 1
            self.MDP.states = value
        return self.MDP
    
    def polciy_improvement(self):
        count = 0
        for i in range(grid_height):
            for j in range(grid_width):
                values = []
                for act in action:
                    row = i + act[0] if (i + act[0] >= 0) and (i + act[0] < grid_height) else i
                    column = j + act[1] if (j + act[1] >= 0) and (j + act[1] < grid_width) else j
                    values.append(self.MDP.states[row, column])
                improved_policy = np.zeros(4)
                values = np.asanyarray(values).round(2)
                maximums = np.where(values == values.max())[0]
                for idx in maximums:
                    improved_policy[idx] = 1/len(maximums)
                self.policy[count] = improved_policy
                count += 1
    
    def show_policy(self):
        current_s = np.chararray((4,4), unicode = True, itemsize = 4)
        count = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if (i == 0 and j == 0) or (i == grid_height-1 and j == grid_width-1):# Terminal State
                    current_s[i, j] = 'T'
                    count += 1
                    continue
                for idx, prob in enumerate(policy[count]):
                    current_s[i, j] += action_string[idx] if prob != 0. else ''              
                count += 1
        print(current_s)
    
    def policy_iteration(self, iteration):
        for m in range(iteration):
            print("@@@@@ {} Iteration ====================".format(m))
            evaluated_GridWorld = self.policy_evaluation()
            print(evaluated_GridWorld.states)
            self.polciy_improvement()

def main():                                           
    print("==================== Policy Iteration ====================")
    grid_world = GridWorldMDP(grid_width, grid_height, reward, discount)
    agent = PolicyIteration(grid_world, action, policy)
    agent.policy_iteration(TOTAL_ITERATION)

if __name__=="__main__":
    main()
