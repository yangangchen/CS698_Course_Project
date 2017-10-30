#####################################################
#       CS 698 Course Project: Tic-tac-toe
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
#   This code trains the AI to learn Tic-tac-toe by reinforcement learning.
#   The methods implemented include:
#   1. Dynamic programming (policy iteration)
#   2. Off-policy TD: Q-Learning
#   The algorithm can be found in Sutton and Barton's "Reinforcement Learning"
#####################################################

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from time import time
from scipy.stats import itemfreq


class TicTacToe:
    ## Initialization of the class
    def __init__(self):
        t0 = time()

        self.states_num = 3 ** 9
        self.actions_num = 9
        self.gamma = 0.9
        self.epsilon = 0  # initial epsilon for epsilon-greedy policy, [0, 1)
        self.alpha0 = 0.5
        self.debug = False

        ## self.states_labels
        self.states_labels = self.evaluate_states_labels(np.array(range(self.states_num)))
        # freq = itemfreq(self.states_labels)
        # print(freq)

        self.illegal_states = np.where(self.states_labels == -1)[0]
        self.legal_states = np.where((self.states_labels == 0) |
                                     (self.states_labels == 1) | (self.states_labels == 2) |
                                     (self.states_labels == 12))[0]
        self.legal_states_num = len(self.legal_states)

        self.ongoing_states = np.where(self.states_labels == 0)[0]
        self.winning1_states = np.where(self.states_labels == 1)[0]
        self.winning2_states = np.where(self.states_labels == 2)[0]
        self.tie_states = np.where(self.states_labels == 12)[0]
        self.start_state = 0

        ## self.states
        self.states = np.array(range(self.states_num))
        self.states[self.illegal_states] = -1

        ## self.equiv_states
        self.equiv_states = -1 * np.ones(self.states_num).astype(int)
        self.equiv_permute = -1 * np.ones(self.states_num).astype(int)
        self.equiv_states[self.legal_states], self.equiv_permute[self.legal_states] = \
            self.evaluate_equivalent_states(self.states[self.legal_states])
        self.unique_equiv_states = np.unique(self.equiv_states[self.legal_states])
        # print(len(self.unique_equiv_states))

        ## self.actionstates_labels
        self.actionstates_labels = self.evaluate_actionstates_labels(self.states) # This is float!

        ## self.reward1, self.reward2
        self.reward1 = np.zeros(self.states_num)
        self.reward1[self.winning1_states] = 1
        self.reward1[self.winning2_states] = -1
        self.reward1[self.tie_states] = 0

        self.reward2 = np.zeros(self.states_num)
        self.reward2[self.winning2_states] = 1
        self.reward2[self.winning1_states] = -1
        self.reward2[self.tie_states] = 0

        ## Value function: self.V1, self.V2
        self.V1 = np.zeros(self.states_num)
        self.V1[self.winning1_states] = 1
        self.V1[self.winning2_states] = -1
        self.V1[self.tie_states] = 0

        self.V2 = np.zeros(self.states_num)
        self.V2[self.winning2_states] = 1
        self.V2[self.winning1_states] = -1
        self.V2[self.tie_states] = 0

        ## Value function: self.Q
        self.Q1 = sps.lil_matrix((self.states_num, self.actions_num))
        self.Q1[self.winning1_states, :] = 1
        self.Q1[self.winning2_states, :] = -1
        self.Q1[self.tie_states, :] = 0

        self.Q2 = sps.lil_matrix((self.states_num, self.actions_num))
        self.Q2[self.winning2_states, :] = 1
        self.Q2[self.winning1_states, :] = -1
        self.Q2[self.tie_states, :] = 0

        ## Policy: self.Pi1, self.Pi2
        Pi = sps.lil_matrix((self.states_num, self.actions_num))
        Pi[self.unique_equiv_states, :] = self.actionstates_labels[self.unique_equiv_states, :]
        self.Pi1 = self.normalize_Prob(Pi)
        self.Pi2 = self.Pi1.copy()

        ## Transition probability: self.tranP1, self.tranP2
        self.tranC1 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))
        self.tranP1 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))

        self.tranC2 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))
        self.tranP2 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))

        ## Transition probability: self.alpha1, self.alpha2
        self.alpha1 = sps.lil_matrix((self.states_num, self.actions_num))  # value function learning rate
        self.alpha2 = sps.lil_matrix((self.states_num, self.actions_num))  # value function learning rate

        t1 = time()
        print('Constructor completed! Time lapse: ' + str(t1-t0))

    ## Load the trained policy (if you want to start playing with smart AI directly)
    def load_policy(self):
        self.Pi1 = sps.lil_matrix(np.load('TicTacToePolicy1.npy'))
        self.Pi2 = sps.lil_matrix(np.load('TicTacToePolicy2.npy'))

    ## Reset the training parameters
    def reset(self):
        self.tranC1 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))
        self.tranP1 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))

        self.tranC2 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))
        self.tranP2 = sps.lil_matrix((self.states_num * self.actions_num, self.states_num))

        self.alpha1 = sps.lil_matrix((self.states_num, self.actions_num))
        self.alpha2 = sps.lil_matrix((self.states_num, self.actions_num))

    ## Show the current configuration of stones
    def show_boardstate(self, state, boardstate):
        if boardstate is None:
            boardstate = self.states_2_boardstates(state)
        show_boardstate = np.array([' '] * 9)
        show_boardstate[boardstate == 1] = 'o'
        show_boardstate[boardstate == 2] = 'x'
        # print('Board configuration')
        print(show_boardstate.reshape((3, 3)))

    ## Convert the representaion of the state from a 9-digit array of (0,1,2) to a single number (0-19683)
    def boardstates_2_states(self, boardstates):
        # "boardstates" can be a matrix, where each row is the single boardstate
        basis = 3 ** np.array(range(9))
        states = boardstates.dot(basis)
        return states

    ## Convert the representaion of the state from a single number (0-19683) to a 9-digit array of (0,1,2)
    def states_2_boardstates(self, states):
        # "boardstates" can be a matrix, where each row is the single boardstate
        if isinstance(states, int) or isinstance(states, np.int64):
            boardstates = np.zeros(9).astype(int)
            for loc in reversed(range(9)):
                boardstates[loc] = states // (3 ** loc)
                states = states % (3 ** loc)
        else:
            boardstates = np.zeros((len(states), 9)).astype(int)
            for loc in reversed(range(9)):
                boardstates[:, loc] = states // (3 ** loc)
                states = states % (3 ** loc)
        return boardstates

    ## Label each state (0-19683)
    # labels:
    # -1: illegal configuration
    # 0: ongoing game
    # 1: player 1 wins
    # 2: player 2 wins
    # 12: player 1 and 2 tie
    def evaluate_states_labels(self, states):
        boardstates = self.states_2_boardstates(states)
        if isinstance(states, int) or isinstance(states, np.int64):
            labels = -1
            if abs(sum(boardstates == 1) - sum(boardstates == 2)) <= 1:
                labels = 0
                win1 = (np.all(boardstates[0:3] == 1) | np.all(boardstates[3:6] == 1) | np.all(boardstates[6:9] == 1) | \
                        np.all(boardstates[[0, 3, 6]] == 1) | np.all(boardstates[[1, 4, 7]] == 1) | \
                        np.all(boardstates[[2, 5, 8]] == 1) | \
                        np.all(boardstates[[0, 4, 8]] == 1) | np.all(boardstates[[2, 4, 6]] == 1))
                win2 = (np.all(boardstates[0:3] == 2) | np.all(boardstates[3:6] == 2) | np.all(boardstates[6:9] == 2) | \
                        np.all(boardstates[[0, 3, 6]] == 2) | np.all(boardstates[[1, 4, 7]] == 2) | \
                        np.all(boardstates[[2, 5, 8]] == 2) | \
                        np.all(boardstates[[0, 4, 8]] == 2) | np.all(boardstates[[2, 4, 6]] == 2))
                if win1:
                    labels = 1 if not win2 and (sum(boardstates == 1) - sum(boardstates == 2) >= 0) else -1
                elif win2:
                    labels = 2 if not win1 and (sum(boardstates == 2) - sum(boardstates == 1) >= 0) else -1
                elif np.all(boardstates != 0):
                    labels = 12
        else:
            labels = -1 * np.ones(len(states)).astype(int)
            legal_indices = abs(np.sum(boardstates == 1, axis=1) - np.sum(boardstates == 2, axis=1)) <= 1
            labels[legal_indices] = 0
            tie_indices = np.all(boardstates != 0, axis=1)
            labels[legal_indices & tie_indices] = 12
            win1_indices = (np.all(boardstates[:, 0:3] == 1, axis=1) | \
                            np.all(boardstates[:, 3:6] == 1, axis=1) | \
                            np.all(boardstates[:, 6:9] == 1, axis=1) | \
                            np.all(boardstates[:, [0, 3, 6]] == 1, axis=1) | \
                            np.all(boardstates[:, [1, 4, 7]] == 1, axis=1) | \
                            np.all(boardstates[:, [2, 5, 8]] == 1, axis=1) | \
                            np.all(boardstates[:, [0, 4, 8]] == 1, axis=1) | \
                            np.all(boardstates[:, [2, 4, 6]] == 1, axis=1))
            win2_indices = (np.all(boardstates[:, 0:3] == 2, axis=1) | \
                            np.all(boardstates[:, 3:6] == 2, axis=1) | \
                            np.all(boardstates[:, 6:9] == 2, axis=1) | \
                            np.all(boardstates[:, [0, 3, 6]] == 2, axis=1) | \
                            np.all(boardstates[:, [1, 4, 7]] == 2, axis=1) | \
                            np.all(boardstates[:, [2, 5, 8]] == 2, axis=1) | \
                            np.all(boardstates[:, [0, 4, 8]] == 2, axis=1) | \
                            np.all(boardstates[:, [2, 4, 6]] == 2, axis=1))
            win1_cond_indices = (np.sum(boardstates == 1, axis=1) - np.sum(boardstates == 2, axis=1) >= 0)
            win2_cond_indices = (np.sum(boardstates == 2, axis=1) - np.sum(boardstates == 1, axis=1) >= 0)
            labels[legal_indices & (win1_indices | win2_indices)] = -1
            labels[legal_indices & win1_indices & (~win2_indices) & win1_cond_indices] = 1
            labels[legal_indices & win2_indices & (~win1_indices) & win2_cond_indices] = 2
        return labels

    ## Label each state-action pair
    # labels:
    # 0: illegal state-action pair
    # 1: legal state-action pair
    def evaluate_actionstates_labels(self, states):
        actionstates_labels = sps.lil_matrix((len(states), self.actions_num))
        indices = (states != -1)
        temp = self.states_2_boardstates(states[indices])
        temp[temp != 0] = -1
        temp += 1
        actionstates_labels[indices, :] = sps.lil_matrix(temp)
        return actionstates_labels

    ## Map a state (0-19683) to its equivalent state under rotations and reflections
    def evaluate_permute_states(self, states, permute, antipermute=False):
        if permute == 0:
            permutelist = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # permutation 1: 90 degree rotation
        elif permute == 1:
            if antipermute:
                permutelist = [2, 5, 8, 1, 4, 7, 0, 3, 6]
            else:
                permutelist = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        # permutation 2: 180 degree rotation
        elif permute == 2:
            permutelist = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        # permutation 3: 270 degree rotation
        elif permute == 3:
            if antipermute:
                permutelist = [6, 3, 0, 7, 4, 1, 8, 5, 2]
            else:
                permutelist = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        # permutation 4: left-right mirror
        elif permute == 4:
            permutelist = [2, 1, 0, 5, 4, 3, 8, 7, 6]
        # permutation 5: up-down mirror
        elif permute == 5:
            permutelist = [6, 7, 8, 3, 4, 5, 0, 1, 2]
        # permutation 6: 90 degree rotation & left-right mirror
        elif permute == 6:
            permutelist = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        # permutation 7: 90 degree rotation & up-down mirror
        elif permute == 7:
            permutelist = [8, 5, 2, 7, 4, 1, 6, 3, 0]

        boardstates = self.states_2_boardstates(states)
        if isinstance(states, int) or isinstance(states, np.int64):
            permute_states = self.boardstates_2_states(boardstates[permutelist])
        else:
            permute_states = self.boardstates_2_states(boardstates[:, permutelist])
        return permute_states

    ## Evaluate the representation of a state (0-19683) inside the minimum set of the unique states
    ## by rotational and reflectional transformations
    def evaluate_equivalent_states(self, states):
        states1 = self.evaluate_permute_states(states, permute=1)
        states2 = self.evaluate_permute_states(states, permute=2)
        states3 = self.evaluate_permute_states(states, permute=3)
        states4 = self.evaluate_permute_states(states, permute=4)
        states5 = self.evaluate_permute_states(states, permute=5)
        states6 = self.evaluate_permute_states(states, permute=6)
        states7 = self.evaluate_permute_states(states, permute=7)
        if isinstance(states, int) or isinstance(states, np.int64):
            # pick the minimum-label equivalent states
            states_all = np.hstack([states, states1, states2, states3,
                                    states4, states5, states6, states7])
            equiv_states = states_all.min()
            equiv_permute = states_all.argmin()
        else:
            # pick the minimum-label equivalent states
            states_all = np.vstack([states, states1, states2, states3,
                                    states4, states5, states6, states7])
            equiv_states = states_all.min(axis=0)
            equiv_permute = states_all.argmin(axis=0)
        return equiv_states, equiv_permute

    ## Map an action (0-9) to its equivalent action under rotations and reflections
    def evaluate_permute_actions(self, actions, permute, antipermute=False):
        if permute == 0:
            permutelist = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ## permutation 1: 90 degree rotation
        elif permute == 1:
            if antipermute:
                permutelist = [6, 3, 0, 7, 4, 1, 8, 5, 2]
            else:
                permutelist = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        ## permutation 2: 180 degree rotation
        elif permute == 2:
            permutelist = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        ## permutation 3: 270 degree rotation
        elif permute == 3:
            if antipermute:
                permutelist = [2, 5, 8, 1, 4, 7, 0, 3, 6]
            else:
                permutelist = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        ## permutation 4: left-right mirror
        elif permute == 4:
            permutelist = [2, 1, 0, 5, 4, 3, 8, 7, 6]
        ## permutation 5: up-down mirror
        elif permute == 5:
            permutelist = [6, 7, 8, 3, 4, 5, 0, 1, 2]
        ## permutation 6: 90 degree rotation & left-right mirror
        elif permute == 6:
            permutelist = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        ## permutation 7: 90 degree rotation & up-down mirror
        elif permute == 7:
            permutelist = [8, 5, 2, 7, 4, 1, 6, 3, 0]
        if isinstance(actions, int):
            permute_actions = permutelist[actions]
        else:
            permutelist = np.array(permutelist)
            permute_actions = permutelist[actions]
        return permute_actions

    ## Normalization of a probability matrix
    def normalize_Prob(self, P):
        factor = np.array(P.tocsr().sum(axis=1).transpose().tolist()[0])
        indices = (factor != 0)
        factor[indices] = (1 / factor[indices])
        P = sps.diags(factor).dot(P)
        return P

    ## Given a player and its state, evaluate its action under its current policy
    def evaluate_action(self, state, player):
        Pi = self.Pi1 if player == 1 else self.Pi2
        equiv_state = self.equiv_states[state]
        equiv_permute = self.equiv_permute[state]
        equiv_action = np.random.choice(range(self.actions_num), p=Pi[equiv_state, :].toarray()[0])
        action = self.evaluate_permute_actions(equiv_action, equiv_permute, antipermute=True)

        if self.debug:
            actions = self.evaluate_permute_actions(range(self.actions_num), equiv_permute, antipermute=False)
            print('policy: ' + str(Pi[equiv_state, actions].toarray()[0]))
            print('equivalent action: ' + str(equiv_action))
            print('action: ' + str(action))
        return state, equiv_state, action, equiv_action

    ## Given a player, its current state and its action, evaluate its next state
    def evaluate_nextstate(self, state, action, player):
        nextstate = state + player * (3 ** action)
        equiv_nextstate = self.equiv_states[nextstate]
        return nextstate, equiv_nextstate

    ## Control the human player's input (0-8)
    def play_input_human(self, player):
        if player == 1:
            action = input("Input: location of stone \'o\', from 0 (top-left) to 8 (bottom-right): ")
        elif player == 2:
            action = input("Input: location of stone \'x\', from 0 (top-left) to 8 (bottom-right): ")
        legal = (action == '0') or (action == '1') or (action == '2') or (action == '3') or\
                (action == '4') or (action == '5') or (action == '6') or (action == '7') or (action == '8')
        return action, legal

    ## The human player plays one step Tic-tac-toe
    def play_one_step_human(self, state, player):
        equiv_state = self.equiv_states[state]
        equiv_permute = self.equiv_permute[state]
        action, legal = self.play_input_human(player)
        while 1:
            if legal:
                action = int(action)
                if self.actionstates_labels[state, action] == 0:
                    print("Illegal input!")
                    action, legal = self.play_input_human(player)
                else:
                    break
            else:
                print("Illegal input!")
                action, legal = self.play_input_human(player)
        equiv_action = self.evaluate_permute_actions(action, equiv_permute)
        nextstate, equiv_nextstate = self.evaluate_nextstate(state, action, player)
        return state, equiv_state, action, equiv_action, nextstate, equiv_nextstate

    ## The AI plays one step Tic-tac-toe
    def play_one_step_AI(self, state, player):
        state, equiv_state, action, equiv_action = self.evaluate_action(state, player)
        nextstate, equiv_nextstate = self.evaluate_nextstate(state, action, player)
        return state, equiv_state, action, equiv_action, nextstate, equiv_nextstate

    ################# Dynamic programming (policy iteration) #################

    ## Record the transition frequency of (state,action -> nextstate)
    def play_record(self, state, action, nextstate, player):
        actionstate = self.actions_num * state + action
        if player == 1:
            self.tranC1[actionstate, nextstate] += 1
        else:
            self.tranC2[actionstate, nextstate] += 1

    ## Play one complete game between human player and AI
    def play_the_whole_game(self, startplayer, humanplayer):
        print("=========== Welcome to Tic-Tac-Toe ===========")
        if humanplayer == 1:
            player1 = 'Yangang'
            player2 = 'AI'
        else:
            player1 = 'AI'
            player2 = 'Yangang'
        self.gamestate = 0
        self.show_boardstate(state=self.gamestate, boardstate=None)
        for turn in range(9):
            if turn % 2 == startplayer - 1:
                print("----- Player 1 (" + str(player1) + ")\'s turn: -----")
                state1 = self.gamestate
                if humanplayer == 1:
                    state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                        self.play_one_step_human(state1, player=1)
                else:
                    state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                        self.play_one_step_AI(state1, player=1)
                self.gamestate = nextstate2
                self.show_boardstate(state=self.gamestate, boardstate=None)
                if turn > 0:  # start recording when turn > 0
                    self.play_record(equiv_state2, equiv_action2, equiv_nextstate2, player=2)
                if self.states_labels[self.gamestate] == 1 or self.states_labels[self.gamestate] == 12:
                    # player 1 terminates the game by winning or tie
                    equiv_nextstate1 = self.equiv_states[self.gamestate]
                    self.play_record(equiv_state1, equiv_action1, equiv_nextstate1, player=1)
                    break
            else:
                print("----- Player 2 (" + str(player2) + ")\'s turn: -----")
                state2 = self.gamestate
                if humanplayer == 2:
                    state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                        self.play_one_step_human(state2, player=2)
                else:
                    state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                        self.play_one_step_AI(state2, player=2)
                self.gamestate = nextstate1
                self.show_boardstate(state=self.gamestate, boardstate=None)
                if turn > 0:  # start recording when turn > 0
                    self.play_record(equiv_state1, equiv_action1, equiv_nextstate1, player=1)
                if self.states_labels[self.gamestate] == 2 or self.states_labels[self.gamestate] == 12:
                    # player 2 terminates the game by winning or tie
                    equiv_nextstate2 = self.equiv_states[self.gamestate]
                    self.play_record(equiv_state2, equiv_action2, equiv_nextstate2, player=2)
                    break
        if self.states_labels[self.gamestate] == 1:
            print("=========== Player 1 (" + str(player1) + ") wins! ===========")
        elif self.states_labels[self.gamestate] == 2:
            print("=========== Player 2 (" + str(player2) + ") wins! ===========")
        elif self.states_labels[self.gamestate] == 12:
            print("=========== Player 1 (" + str(player1) + ") and 2 (" + str(player2) + ") tie! ===========")

    ## Play one complete game between two AIs
    def play_random_games(self, player, startplayer):
        self.gamestate = 0
        for turn in range(9):
            if turn % 2 == startplayer - 1:
                state1 = self.gamestate
                state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                    self.play_one_step_AI(state1, player=1)
                self.gamestate = nextstate2
                if (player == 2 or player == 12) and turn > 0:  # start recording when turn > 0
                    self.play_record(equiv_state2, equiv_action2, equiv_nextstate2, player=2)
                if self.states_labels[self.gamestate] == 1 or self.states_labels[self.gamestate] == 12:
                    # player 1 terminates the game by winning or tie
                    if player == 1 or player == 12:
                        equiv_nextstate1 = self.equiv_states[self.gamestate]
                        self.play_record(equiv_state1, equiv_action1, equiv_nextstate1, player=1)
                    break
            else:
                state2 = self.gamestate
                state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                    self.play_one_step_AI(state2, player=2)
                self.gamestate = nextstate1
                if (player == 1 or player == 12) and turn > 0:  # start recording when turn > 0
                    self.play_record(equiv_state1, equiv_action1, equiv_nextstate1, player=1)
                if self.states_labels[self.gamestate] == 2 or self.states_labels[self.gamestate] == 12:
                    # player 2 terminates the game by winning or tie
                    if player == 2 or player == 12:
                        equiv_nextstate2 = self.equiv_states[self.gamestate]
                        self.play_record(equiv_state2, equiv_action2, equiv_nextstate2, player=2)
                    break

    ## Date generation for policy iteration
    ## In particular, statistically evaluation of p(s'|s,a)
    def data_generation(self, player, max_episode=3000):
        t0 = time()

        for episode in range(max_episode):
            if episode % 100 == 0:
                print(episode)
            self.play_random_games(player, startplayer=1)
            self.play_random_games(player, startplayer=2)

        if player == 1 or player == 12:
            self.tranP1 = self.normalize_Prob(self.tranC1)
        if player == 2 or player == 12:
            self.tranP2 = self.normalize_Prob(self.tranC2)

        t1 = time()
        print('Data generation completed! Time lapse: ' + str(t1 - t0))

    ## Update the value function V from the action-value function Q
    def updateQ_fromV(self, player):
        if player == 1:
            reward = self.reward1
            tranP = self.tranP1
            V = self.V1
            Q = self.Q1
        else:
            reward = self.reward2
            tranP = self.tranP2
            V = self.V2
            Q = self.Q2
        for state in self.unique_equiv_states:
            tranP_state = tranP[self.actions_num * state:self.actions_num * (state + 1), :]
            Q[state, :] = tranP_state.dot(reward) + tranP_state.dot(self.gamma * V)
        if player == 1:
            self.Q1 = Q
        else:
            self.Q2 = Q

    ## Update the action-value function Q from the value function V
    def updateV_fromQ(self, player):
        if player == 1:
            Pi = self.Pi1
            V = self.V1
            Q = self.Q1
        else:
            Pi = self.Pi2
            V = self.V2
            Q = self.Q2
        V[self.unique_equiv_states] = np.array(np.sum(
            Pi[self.unique_equiv_states, :].multiply(Q[self.unique_equiv_states, :]), axis=1
        ).transpose().tolist()[0])
        if player == 1:
            self.V1 = V
        else:
            self.V2 = V

    ## Update the policy at a given state
    def update_policy(self, state, player, greedy=True):
        if player == 1:
            Pi = self.Pi1
            Q = self.Q1
        else:
            Pi = self.Pi2
            Q = self.Q2

        action_indices = (self.actionstates_labels[state, :].toarray()[0] == 1)
        actions_num = np.sum(action_indices)
        epsilon = 0 if greedy else self.epsilon
        Pi[state, :] = 0
        Pi[state, action_indices] = epsilon

        optimal_Q_state = Q[state, action_indices].toarray()[0].max()
        optimal_action_indices = action_indices & (abs(Q[state, :].toarray()[0] - optimal_Q_state) <= 1e-6)
        optimal_actions_num = np.sum(optimal_action_indices)
        Pi[state, optimal_action_indices] = epsilon + (1 - epsilon * actions_num) / optimal_actions_num

        if player == 1:
            self.Pi1 = Pi
        else:
            self.Pi2 = Pi

    ## Construct the linear system of Bellman equations
    def linear_system(self, player):
        self.A = sps.lil_matrix((self.states_num, self.states_num))
        if player == 1:
            self.F = self.reward1.copy()
            reward = self.reward1
            tranP = self.tranP1
            Pi = self.Pi1
        else:
            self.F = self.reward2.copy()
            reward = self.reward2
            tranP = self.tranP2
            Pi = self.Pi2
        for state in self.unique_equiv_states:
            self.A[state, state] = 1
            if self.states_labels[state] == 0:
                tranP_state = tranP[self.actions_num * state:self.actions_num * (state + 1), :]
                self.A[state, :] = - self.gamma * Pi[state, :].dot(tranP_state)
                self.A[state, state] = 1 + self.A[state, state]
                self.F[state] = Pi[state, :].dot(tranP_state.dot(reward))

    ## Dynamic programming (policy iteration): The step of policy evaluation
    def policy_evaluation(self, player):
        t0 = time()

        if player == 1:
            V = self.V1
        else:
            V = self.V2

        self.linear_system(player)
        self.A = self.A[self.unique_equiv_states, :][:, self.unique_equiv_states]
        self.F = self.F[self.unique_equiv_states]
        V[self.unique_equiv_states] = spsl.spsolve(self.A.tocsr(), self.F)

        if player == 1:
            self.V1 = V
        else:
            self.V2 = V

        t1 = time()
        print('Policy evaluation completed! Time lapse: ' + str(t1-t0))

    ## Dynamic programming (policy iteration): The step of policy improvement
    def policy_improvement(self, player, greedy=True):
        t0 = time()

        self.updateQ_fromV(player)
        for state in self.unique_equiv_states:
            if self.states_labels[state] == 0:
                self.update_policy(state, player, greedy)

        t1 = time()
        print('Policy improvement completed! Time lapse: ' + str(t1 - t0))

    ## Dynamic programming (policy iteration): Complete process
    def policy_iteration(self, player, max_iter=100):
        if player == 1:
            V = self.V1
        else:
            V = self.V2

        for iter in range(max_iter):
            V0 = V.copy()
            self.policy_evaluation(player)
            self.policy_improvement(player, greedy=False)
            converge = np.sum(abs(V - V0))
            print(converge)
            if converge <= 1e-6:
                print('Policy iteration succeeds in ' + str(iter) + ' iterations!')
                break

        if player == 1:
            self.V1 = V
            np.save('TicTacToePolicy1.npy', self.Pi1.todense())
        else:
            self.V2 = V
            np.save('TicTacToePolicy2.npy', self.Pi2.todense())

    ## Dynamic programming (policy iteration)
    ## Complete training process
    def policy_training(self):
        self.data_generation(player=2, max_episode=2000)
        self.policy_iteration(player=2)

    ################# Q-Learning(0) Method #################

    ## Update the action-value function for each step of Q-Learning
    def QLearning_updateQ(self, state, action, nextstate, player, rate_decrease=True):
        if player == 1:
            Q = self.Q1
            alpha = self.alpha1
            reward = self.reward1
        else:
            Q = self.Q2
            alpha = self.alpha2
            reward = self.reward2

        if self.states_labels[nextstate] == 1 or self.states_labels[nextstate] == 2 \
                or self.states_labels[nextstate] == 12:
            optimal_Q_nextstate = reward[nextstate]
        else:
            action_indices = (self.actionstates_labels[nextstate, :].toarray()[0] == 1)
            optimal_Q_nextstate = Q[nextstate, action_indices].toarray()[0].max()

        Q[state, action] += self.alpha0 / (alpha[state, action] + 1) * (
            reward[nextstate] + self.gamma * optimal_Q_nextstate - Q[state, action])

        if rate_decrease:
            alpha[state, action] += 1

        if player == 1:
            self.Q1 = Q
            self.alpha1 = alpha
        else:
            self.Q2 = Q
            self.alpha2 = alpha

    ## Play one complete game between human player and AI
    def QLearning_play_the_whole_game(self, startplayer, humanplayer, rate_decrease1=True, rate_decrease2=True):
        print("=========== Welcome to Tic-Tac-Toe ===========")
        if humanplayer == 1:
            player1 = 'Yangang'
            player2 = 'AI'
        else:
            player1 = 'AI'
            player2 = 'Yangang'
        self.gamestate = 0
        self.show_boardstate(state=self.gamestate, boardstate=None)
        for turn in range(9):
            if turn % 2 == startplayer - 1:
                print("----- Player 1 (" + str(player1) + ")\'s turn: -----")
                state1 = self.gamestate
                equiv_state1 = self.equiv_states[self.gamestate]  # Be careful
                self.update_policy(equiv_state1, player=1, greedy=False)  # Be careful
                if humanplayer == 1:
                    state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                        self.play_one_step_human(state1, player=1)
                else:
                    state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                        self.play_one_step_AI(state1, player=1)
                self.gamestate = nextstate2
                self.show_boardstate(state=self.gamestate, boardstate=None)
                if turn > 0:  # start recording when turn > 0
                    self.QLearning_updateQ(equiv_state2, equiv_action2, equiv_nextstate2,
                                           player=2, rate_decrease=rate_decrease2)  # Be careful
                if self.states_labels[self.gamestate] == 1 or self.states_labels[self.gamestate] == 12:
                    # player 1 terminates the game by winning or tie
                    equiv_nextstate1 = self.equiv_states[self.gamestate]
                    self.QLearning_updateQ(equiv_state1, equiv_action1, equiv_nextstate1,
                                           player=1, rate_decrease=rate_decrease1)  # Be careful
                    break
            else:
                print("----- Player 2 (" + str(player2) + ")\'s turn: -----")
                state2 = self.gamestate
                equiv_state2 = self.equiv_states[self.gamestate]  # Be careful
                self.update_policy(equiv_state2, player=2, greedy=False)  # Be careful
                if humanplayer == 2:
                    state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                        self.play_one_step_human(state2, player=2)
                else:
                    state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                        self.play_one_step_AI(state2, player=2)
                self.gamestate = nextstate1
                self.show_boardstate(state=self.gamestate, boardstate=None)
                if turn > 0:  # start recording when turn > 0
                    self.QLearning_updateQ(equiv_state1, equiv_action1, equiv_nextstate1,
                                           player=1, rate_decrease=rate_decrease1)  # Be careful
                if self.states_labels[self.gamestate] == 2 or self.states_labels[self.gamestate] == 12:
                    # player 2 terminates the game by winning or tie
                    equiv_nextstate2 = self.equiv_states[self.gamestate]
                    self.QLearning_updateQ(equiv_state2, equiv_action2, equiv_nextstate2,
                                           player=2, rate_decrease=rate_decrease2)  # Be careful
                    break
        if self.states_labels[self.gamestate] == 1:
            print("=========== Player 1 (" + str(player1) + ") wins! ===========")
        elif self.states_labels[self.gamestate] == 2:
            print("=========== Player 2 (" + str(player2) + ") wins! ===========")
        elif self.states_labels[self.gamestate] == 12:
            print("=========== Player 1 (" + str(player1) + ") and 2 (" + str(player2) + ") tie! ===========")

        # self.updateV_fromQ(player=1)
        # self.updateV_fromQ(player=2)

    ## Play one complete game between two AIs
    def QLearning_play_random_games(self, startplayer, rate_decrease1=True, rate_decrease2=True):
        self.gamestate = 0
        for turn in range(9):
            if turn % 2 == startplayer - 1:
                state1 = self.gamestate
                equiv_state1 = self.equiv_states[self.gamestate]  # Be careful
                self.update_policy(equiv_state1, player=1, greedy=False)  # Be careful
                state1, equiv_state1, action1, equiv_action1, nextstate2, equiv_nextstate2 = \
                    self.play_one_step_AI(state1, player=1)
                self.gamestate = nextstate2
                if turn > 0:  # start recording when turn > 0
                    self.QLearning_updateQ(equiv_state2, equiv_action2, equiv_nextstate2,
                                           player=2, rate_decrease=rate_decrease2)  # Be careful
                if self.states_labels[self.gamestate] == 1 or self.states_labels[self.gamestate] == 12:
                    # player 1 terminates the game by winning or tie
                    equiv_nextstate1 = self.equiv_states[self.gamestate]
                    self.QLearning_updateQ(equiv_state1, equiv_action1, equiv_nextstate1,
                                           player=1, rate_decrease=rate_decrease1)  # Be careful
                    break
            else:
                state2 = self.gamestate
                equiv_state2 = self.equiv_states[self.gamestate]  # Be careful
                self.update_policy(equiv_state2, player=2, greedy=False)  # Be careful
                state2, equiv_state2, action2, equiv_action2, nextstate1, equiv_nextstate1 = \
                    self.play_one_step_AI(state2, player=2)
                self.gamestate = nextstate1
                if turn > 0:  # start recording when turn > 0
                    self.QLearning_updateQ(equiv_state1, equiv_action1, equiv_nextstate1,
                                           player=1, rate_decrease=rate_decrease1)  # Be careful
                if self.states_labels[self.gamestate] == 2 or self.states_labels[self.gamestate] == 12:
                    # player 2 terminates the game by winning or tie
                    equiv_nextstate2 = self.equiv_states[self.gamestate]
                    self.QLearning_updateQ(equiv_state2, equiv_action2, equiv_nextstate2,
                                           player=2, rate_decrease=rate_decrease2)  # Be careful
                    break

        # self.updateV_fromQ(player=1)
        # self.updateV_fromQ(player=2)

    ## Q-Learning(0) method:
    ## Complete training process
    def QLearning_training(self, max_episode=3000):
        t0 = time()
        for episode in range(max_episode):
            if episode % 100 == 0:
                print(episode)
                V10 = self.V1.copy()
                V20 = self.V2.copy()

            ## TD QLearning in each episode
            rate_decrease1 = False
            rate_decrease2 = True
            self.QLearning_play_random_games(startplayer=1, rate_decrease1=rate_decrease1, rate_decrease2=rate_decrease2)
            self.QLearning_play_random_games(startplayer=2, rate_decrease1=rate_decrease1, rate_decrease2=rate_decrease2)

            if episode % 100 == 99:
                self.updateV_fromQ(player=1)
                self.updateV_fromQ(player=2)
                converge1 = np.sum(abs(self.V1 - V10))
                converge2 = np.sum(abs(self.V2 - V20))
                print([converge1, converge2])
                if converge1 <= 1e-6 and converge2 <= 1e-6:
                    print('TD(0) Q-Learning succeeds in ' + str(episode + 1) + ' episodes!')
                    break

        t1 = time()
        print('time = ' + str(t1-t0))
        # np.save('TicTacToePolicy1.npy', self.Pi1.todense())
        # np.save('TicTacToePolicy2.npy', self.Pi2.todense())
        # np.save('TicTacToeV1.npy', self.V1.todense())
        # np.save('TicTacToeV2.npy', self.V2.todense())
        # np.save('TicTacToeQ1.npy', self.Q1.todense())
        # np.save('TicTacToeQ2.npy', self.Q2.todense())
