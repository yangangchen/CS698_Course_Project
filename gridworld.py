#####################################################
#       CS 698 Course Project: GridWorld
#               Yangang Chen
#           University of Waterloo
#              April 15, 2017
# 
# Copyright (C) 2017  Yangang Chen
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# 
#   This code implements GridWorld. The example of GridWorld is from
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
#   The methods implemented include:
#   1. Dynamic programming (policy iteration)
#   2. Monte Carlo
#   3. On-policy TD: Sarsa
#   4. Off-policy TD: Q-Learning
#   5. On-policy TD(lambda): Sarsa(lambda)
#   6. Off-policy TD(lambda): Q-Learning(lambda)
#   The algorithm can be found in Sutton and Barton's "Reinforcement Learning"
#####################################################

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


class GridWorld:
    ## Initialization of the class
    def __init__(self, deadend=True):
        self.state_num = 12 ** 2
        self.action_num = 4
        self.gamma = 0.9
        self.epsilon = 0  # initial epsilon for epsilon-greedy policy, [0, 1)
        self.alpha0 = 0.2

        self.end_point = 78
        self.dead_points = np.hstack([range(12), range(12, 132, 12), range(23, 143, 12), range(132, 144),
                                      range(38, 42), range(43, 46), range(53, 102, 12)])
        self.live_points = np.array(list(set(range(144)) - {self.end_point} - set(self.dead_points)))
        self.start_point = self.live_points[0]

        self.penalty_points = [52, 66, 67, 79, 81, 93, 100, 102, 103]

        self.nextstate = np.vstack([np.array(range(144)) - 12, np.array(range(144)) + 12,
                                    np.array(range(144)) - 1, np.array(range(144)) + 1]).transpose()
        for state in range(144):
            if state in self.dead_points:
                self.nextstate[state, :] = state
            elif state == self.end_point:
                self.nextstate[state, :] = self.dead_points[0] if deadend else self.start_point
            else:
                for action in range(self.action_num):
                    if self.nextstate[state, action] in self.dead_points:
                        self.nextstate[state, action] = state

        self.reward = np.zeros(self.state_num)
        self.reward[self.live_points] = -1e-4
        self.reward[self.end_point] = 1
        self.reward[self.penalty_points] = -1

        self.Count = sps.lil_matrix((self.state_num * self.action_num, self.state_num))
        self.Prob = sps.lil_matrix((self.state_num * self.action_num, self.state_num))
        self.R = sps.lil_matrix((self.state_num * self.action_num, self.state_num))

        self.Pi = np.zeros((self.state_num, self.action_num))
        self.Pi[self.live_points, :] = 1 / self.action_num
        self.Pi[self.end_point, :] = 0
        self.Pi[self.end_point, 0] = 1

        self.V = np.zeros(self.state_num)
        self.Q = np.zeros((self.state_num, self.action_num))

    ## Show the GridWorld value function on the screen
    def show_value(self):
        print('Value')
        print(self.V.reshape(12, 12)[1:11, 1:11])

    ## Show the GridWorld policy on the screen
    # u: up
    # d: down
    # l: left
    # r: right
    def show_policy(self):
        print('Policy')
        Pishow = [''] * self.state_num
        for state in range(self.state_num):
            if state in self.dead_points:
                Pishow[state] = '    '
            elif state == self.end_point:
                Pishow[state] = ' ** '
            else:
                policy = ''
                optimalPi = self.Pi[state, :].max()
                policy += 'u' if self.Pi[state, 0] == optimalPi else ' '
                policy += 'd' if self.Pi[state, 1] == optimalPi else ' '
                policy += 'l' if self.Pi[state, 2] == optimalPi else ' '
                policy += 'r' if self.Pi[state, 3] == optimalPi else ' '
                Pishow[state] = policy
        Pishow = np.array(Pishow)
        print(Pishow.reshape(12, 12)[1:11, 1:11])

    ################# Dynamic programming (policy iteration) #################

    ## Generate the exact transition probability matrix p(s'|s,a)
    def Prob_exact(self):
        for state in range(self.state_num):
            if state not in self.dead_points:
                for action in range(self.action_num):
                    nextstate = self.nextstate[state, action]
                    self.Prob_update(state, action, nextstate)

    ## Update the transition probability matrix p(s'|s,a)
    def Prob_update(self, state, action, nextstate):
        state_action = self.action_num * state + action
        self.Count[state_action, nextstate] += 1
        self.Prob[state_action, :] = self.Count[state_action, :] / np.sum(self.Count[state_action, :])
        self.R[state_action, :] = self.reward[state]

    ## Update the value function V from the action-value function Q
    def updateQ_fromV(self):
        # Q0 = self.Q.copy()
        self.Q = np.array(np.sum(self.Prob.multiply(self.R),axis=1).transpose().tolist()[0])\
                 + self.Prob.dot(self.gamma * self.V)
        self.Q = self.Q.reshape((self.state_num, self.action_num))
        self.Q[self.dead_points] = 0
        # print('dQ = '+str(np.sum(abs(self.Q-Q0))))

    ## Update the action-value function Q from the value function V
    def updateV_fromQ(self):
        self.V = np.sum(self.Pi * self.Q, axis=1)
        # self.V = self.Pi.dot(self.Q.transpose()).diagonal()

    ## Update the policy at a given state
    def update_policy(self, state, greedy=True):
        optimalQ = self.Q[state, :].max()
        optimalactions = self.Q[state, :] == optimalQ
        epsilon = 0 if greedy else self.epsilon
        self.Pi[state, :] = epsilon / self.action_num
        self.Pi[state, optimalactions] += (1 - epsilon) / sum(optimalactions)

    ## Construct the linear system of Bellman equations
    def linear_system(self):
        self.A = sps.identity(n=self.state_num).tolil()
        self.F = np.zeros(self.state_num)
        for state in range(self.state_num):
            Prob_state = self.Prob[self.action_num * state:self.action_num * (state + 1), :]
            self.A[state, :] -= self.gamma * Prob_state.transpose().dot(self.Pi[state, :])
            self.F[state] = np.sum(Prob_state.transpose().dot(self.Pi[state, :])) * self.reward[state]
            # Note that the reward is only a function of the state!

    ## Dynamic programming (policy iteration): The step of policy evaluation
    def policy_evaluation(self):
        self.linear_system()
        self.V[:] = spsl.spsolve(self.A.tocsr(), self.F)

    ## Dynamic programming (policy iteration): The step of policy improvement
    def policy_improvement(self, greedy=True):
        self.updateQ_fromV()
        for state in range(self.state_num):
            if state not in self.dead_points:
                self.update_policy(state, greedy)

    ## Dynamic programming (policy iteration): Complete process
    def policy_iteration(self, max_iter=100):
        self.Prob_exact()
        for iter in range(max_iter):
            V0 = self.V.copy()
            self.policy_evaluation()
            self.policy_improvement(greedy=True)
            converge = np.sum(abs(self.V - V0))
            print(converge)
            if converge <= 1e-6:
                print('Policy iteration succeeds in ' + str(iter) + ' iterations!')
                break

    ################# Monte Carlo methods #################

    ## One episode Monte Carlo simulation
    def MonteCarlo_episode(self, initial_state, initial_action, TotalReturn, TotalVisits, max_T=1000):
        state_list = []
        action_list = []
        reward_list = []

        for t in range(max_T):
            ## (state, action)
            if t == 0:
                state = initial_state
                action = initial_action
            else:
                state = self.nextstate[state, action]
                action = np.random.choice(range(self.action_num), p=self.Pi[state, :])
            ## reward
            reward = self.reward[state]
            ## appending
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            ## terminal of the episode
            if state == self.end_point:
                break

        T = len(state_list)
        visited_flag = np.zeros((self.state_num, self.action_num)).astype(int)
        # if T < max_T: ## valid terminated episode
        if True:  ## all terminated episode
            total_reward_list = np.array(reward_list)
            for t in reversed(range(T - 1)):
                total_reward_list[t] += self.gamma * total_reward_list[t + 1]

            for t in range(T):
                state = state_list[t]
                action = action_list[t]
                if not visited_flag[state, action]:
                    visited_flag[state, action] = 1
                    TotalReturn[state, action] += total_reward_list[t]
                    TotalVisits[state, action] += 1
                    self.Q[state, action] = TotalReturn[state, action] / TotalVisits[state, action]

        self.updateV_fromQ()

        ## Update policy
        for t in range(T):
            state = state_list[t]
            self.update_policy(state, greedy=False)

    ## Complete Monte Carlo simulation
    def MonteCarlo(self, max_episode=1000000, max_T=1000):
        TotalReturn = np.zeros((self.state_num, self.action_num))
        TotalVisits = np.zeros((self.state_num, self.action_num)).astype(int)

        for episode in range(max_episode):
            if episode % 1000 == 0:
                print(episode)
                V0 = self.V.copy()

            ## Initial (state, action): for all (state, action)
            initial_state = self.live_points[episode // self.action_num % len(self.live_points)]
            initial_action = episode % self.action_num
            # initial_state = np.random.choice(self.live_points,
            #                                  p=1 / len(self.live_points) * np.ones(len(self.live_points)))
            # initial_action = np.random.choice(range(self.action_num),
            #                                   p=1 / self.action_num * np.ones(self.action_num))
            # print([initial_state, initial_action])

            ## Monte Carlo in each episode
            self.MonteCarlo_episode(initial_state, initial_action, TotalReturn, TotalVisits, max_T)

            if episode % 1000 == 999:
                converge = np.sum(abs(self.V - V0))
                print(converge)
                if converge <= 1e-6:
                    print('Monte Carlo succeeds in ' + str(episode + 1) + ' episodes!')
                    break

    ################# On-policy TD methods: Sarsa(0) #################

    ## One episode Sarsa simulation
    def TD_Sarsa_episode(self, initial_state, max_T=1000, rate_decrease=True):
        state = initial_state
        self.update_policy(state, greedy=False)
        action = np.random.choice(range(self.action_num), p=self.Pi[state, :])

        for t in range(max_T):
            reward = self.reward[state]

            nextstate = self.nextstate[state, action]
            self.update_policy(nextstate, greedy=False)
            nextaction = np.random.choice(range(self.action_num), p=self.Pi[nextstate, :])

            self.Q[state, action] += self.alpha0 / self.alpha[state, action] * (
                reward + self.gamma * self.Q[nextstate, nextaction] - self.Q[state, action])
            if rate_decrease:
                self.alpha[state, action] += 1

            if state == self.end_point:
                break

            state = nextstate
            action = nextaction

        self.updateV_fromQ()

    ## Complete Sarsa simulation
    def TD_Sarsa(self, max_episode=100000, max_T=1000):
        self.alpha = np.ones((self.state_num, self.action_num)).astype(int)  # value function learning rate
        for episode in range(max_episode):
            if episode % 1000 == 0:
                print(episode)
                V0 = self.V.copy()

            ## Initial state: for all states
            initial_state = self.live_points[episode % len(self.live_points)]
            # initial_state = np.random.choice(self.live_points,
            #                                  p=1 / len(self.live_points) * np.ones(len(self.live_points)))

            ## TD Sarsa in each episode
            rate_decrease = False
            # rate_decrease = (episode >= max_episode // 10)
            self.TD_Sarsa_episode(initial_state,
                                  max_T=max_T, rate_decrease=rate_decrease)

            if episode % 1000 == 999:
                converge = np.sum(abs(self.V - V0))
                print(converge)
                if converge <= 1e-6:
                    print('TD(0) Sarsa succeeds in ' + str(episode + 1) + ' episodes!')
                    break

    ################# Off-policy TD methods: Q-Learning(0) #################

    ## One episode Q-Learning simulation
    def TD_QLearning_episode(self, initial_state, max_T=1000, rate_decrease=True):
        state = initial_state

        for t in range(max_T):
            self.update_policy(state, greedy=False)
            action = np.random.choice(range(self.action_num), p=self.Pi[state, :])
            reward = self.reward[state]
            nextstate = self.nextstate[state, action]

            self.Q[state, action] += self.alpha0 / self.alpha[state, action] * (
                reward + self.gamma * self.Q[nextstate, :].max() - self.Q[state, action])
            if rate_decrease:
                self.alpha[state, action] += 1

            if state == self.end_point:
                break

            state = nextstate

        self.updateV_fromQ()

    ## Complete Q-Learning simulation
    def TD_QLearning(self, max_episode=100000, max_T=1000):
        self.alpha = np.ones((self.state_num, self.action_num)).astype(int)  # value function learning rate
        for episode in range(max_episode):
            if episode % 1000 == 0:
                print(episode)
                V0 = self.V.copy()

            ## Initial state: for all states
            initial_state = self.live_points[episode % len(self.live_points)]
            # initial_state = np.random.choice(self.live_points,
            #                                  p=1 / len(self.live_points) * np.ones(len(self.live_points)))

            ## TD QLearning in each episode
            rate_decrease = False
            # rate_decrease = (episode >= max_episode // 10)
            self.TD_QLearning_episode(initial_state,
                                      max_T=max_T, rate_decrease=rate_decrease)

            if episode % 1000 == 999:
                converge = np.sum(abs(self.V - V0))
                print(converge)
                if converge <= 1e-6:
                    print('TD(0) Q-Learning succeeds in ' + str(episode + 1) + ' episodes!')
                    break

    ################# On-policy TD methods: Sarsa(lambda) #################

    ## One episode Sarsa simulation
    def TDlambda_Sarsa_episode(self, initial_state, Lambda=0, max_T=1000, rate_decrease=True):
        state = initial_state
        self.update_policy(state, greedy=False)
        action = np.random.choice(range(self.action_num), p=self.Pi[state, :])

        for t in range(max_T):
            reward = self.reward[state]

            nextstate = self.nextstate[state, action]
            self.update_policy(nextstate, greedy=False)
            nextaction = np.random.choice(range(self.action_num), p=self.Pi[nextstate, :])

            delta = reward + self.gamma * self.Q[nextstate, nextaction] - self.Q[state, action]
            self.e[state, action] += 1
            self.Q += self.alpha0 / self.alpha * delta * self.e
            if rate_decrease:
                self.alpha[state, action] += 1
            self.e *= self.gamma * Lambda

            if state == self.end_point:
                break

            state = nextstate
            action = nextaction

        self.updateV_fromQ()

    ## Complete Sarsa simulation
    def TDlambda_Sarsa(self, Lambda=0, max_episode=100000, max_T=1000):
        self.alpha = np.ones((self.state_num, self.action_num)).astype(int)  # value function learning rate
        self.e = np.zeros((self.state_num, self.action_num))
        for episode in range(max_episode):
            if episode % 1000 == 0:
                print(episode)
                V0 = self.V.copy()

            ## Initial state: for all states
            initial_state = self.live_points[episode % len(self.live_points)]
            # initial_state = np.random.choice(self.live_points,
            #                                  p=1 / len(self.live_points) * np.ones(len(self.live_points)))

            ## TD Sarsa in each episode
            rate_decrease = False
            # rate_decrease = (episode >= max_episode // 10)
            self.TDlambda_Sarsa_episode(initial_state,
                                        Lambda=Lambda, max_T=max_T, rate_decrease=rate_decrease)

            if episode % 1000 == 999:
                converge = np.sum(abs(self.V - V0))
                print(converge)
                if converge <= 1e-6:
                    print('TD(' + str(Lambda) + ') Sarsa succeeds in ' + str(episode + 1) + ' episodes!')
                    break

    ################# Off-policy TD methods: Q-Learning(lambda) #################

    ## One episode Q-Learning simulation
    def TDlambda_QLearning_episode(self, initial_state, Lambda=0, max_T=1000, rate_decrease=True):
        state = initial_state
        self.update_policy(state, greedy=False)
        action = np.random.choice(range(self.action_num), p=self.Pi[state, :])

        for t in range(max_T):
            reward = self.reward[state]

            nextstate = self.nextstate[state, action]
            self.update_policy(nextstate, greedy=False)
            nextaction = np.random.choice(range(self.action_num), p=self.Pi[nextstate, :])

            optimalaction = self.Q[nextstate, :].argmax()
            if self.Q[nextstate, nextaction] == self.Q[nextstate, optimalaction]:
                optimalaction = nextaction

            delta = reward + self.gamma * self.Q[nextstate, optimalaction] - self.Q[state, action]
            self.e[state, action] += 1
            self.Q += self.alpha0 / self.alpha * delta * self.e
            if rate_decrease:
                self.alpha[state, action] += 1
            self.e *= self.gamma * Lambda if nextaction == optimalaction else 0

            if state == self.end_point:
                break

            state = nextstate
            action = nextaction

        self.updateV_fromQ()

    ## Complete Q-Learning simulation
    def TDlambda_QLearning(self, Lambda=0, max_episode=100000, max_T=1000):
        self.alpha = np.ones((self.state_num, self.action_num)).astype(int)  # value function learning rate
        self.e = np.zeros((self.state_num, self.action_num))
        for episode in range(max_episode):
            if episode % 1000 == 0:
                print(episode)
                V0 = self.V.copy()

            ## Initial state: for all states
            initial_state = self.live_points[episode % len(self.live_points)]
            # initial_state = np.random.choice(self.live_points,
            #                                  p=1 / len(self.live_points) * np.ones(len(self.live_points)))

            ## TD QLearning in each episode
            rate_decrease = False
            # rate_decrease = (episode >= max_episode // 10)
            self.TDlambda_QLearning_episode(initial_state,
                                            Lambda=Lambda, max_T=max_T, rate_decrease=rate_decrease)

            if episode % 1000 == 999:
                converge = np.sum(abs(self.V - V0))
                print(converge)
                if converge <= 1e-6:
                    print('TD(' + str(Lambda) + ') Q-Learning succeeds in ' + str(episode + 1) + ' episodes!')
                    break
