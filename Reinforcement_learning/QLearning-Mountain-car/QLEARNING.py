# coding:utf-8

import numpy as np
import random


class qlearning:

    def __init__(self, num_states, actions, alpha, gamma, epsilon, n_episodes=1000,):
        self.q = np.zeros((num_states[0], num_states[1], len(actions)))

        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q[state[0], state[1], action]


    def choose_action(self, state): # epsilon greedy strategy
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_q(state, act) for act in self.actions]
            max_q = max(q)
            if q.count(max_q) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_q]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_q)]
        return action

    def learn(self, state1, action, state2, reward):
        old_q = self.get_q(state1, action)
        if old_q == 0:  

            self.q[state1[0], state1[1], action] = reward

        next_max_q = max([self.get_q(state2, a) for a in self.actions])
        self.q[state1[0], state1[1], action] = old_q + self.alpha * (
                    reward + self.gamma * next_max_q - old_q)

