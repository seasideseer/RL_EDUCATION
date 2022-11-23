# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:01:49 2022

@author: seasideseer
"""

# %% Import Modules

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.express as px

# %% Class


class Arm():

    def __init__(self, action, mean, sd):
        self.action = action
        self.mean = mean
        self.sd = sd

    def reward(self):
        r = np.random.normal(loc=self.mean,
                             scale=self.sd,
                             size=1)[0]
        return(r)


class MultiArmedBandits():

    def __init__(self,
                 arm_s):
        self.arm_s = np.array(arm_s)
        self.k = len(arm_s)
        self.action_s = self.get_actions()

    def get_actions(self):
        action_s = []
        for arm in self.arm_s:
            action_s.append(arm.action)
        action_s = np.array(action_s)
        return(action_s)

    def step(self, action):
        arm = self.arm_s[self.action_s == action][0]
        reward = arm.reward()
        return(reward)


class EpsGreedy():

    def __init__(self):
        pass

    def algorithm(self, env, eps=0.2, plotFig=True):
        q_dict = {}
        for action in env.action_s:
            q_dict[action] = 0

        N_dict = {}
        for action in env.action_s:
            N_dict[action] = 0

        action_record = []
        reward_record = []

        for i in np.r_[0:1000]:
            rdn = np.random.uniform(0, 1, 1)[0]
            if rdn <= eps:
                action = np.random.choice(env.action_s)
            else:
                action = env.action_s[list(q_dict.values()) ==
                                      np.max(list(q_dict.values()))][0]
            reward_now = env.step(action)
            action_record.append(action)
            reward_record.append(reward_now)

            N_dict[action] += 1
            q_dict[action] = q_dict[action] + \
                (reward_now-q_dict[action])/N_dict[action]

        record = pd.DataFrame({'N': np.r_[0:len(action_record)],
                               'action': action_record,
                               'reward': reward_record})

        if plotFig:
            fig = px.scatter(record,
                             x="N",
                             y="reward",
                             color="action",
                             title="epsilon : "+str(eps))
            plot(fig)

        return(q_dict, N_dict, record)

# %% Example


arm0 = Arm(0, 10, 5)
arm1 = Arm(1, 10, 21)
arm2 = Arm(2, 50, 11)

mab = MultiArmedBandits(arm_s=[arm0, arm1, arm2])
eg = EpsGreedy()

for e in np.r_[0.01, 0.1, 0.5]:
    q_dict, N_dict, record = eg.algorithm(mab, eps=e)
