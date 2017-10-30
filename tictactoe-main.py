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

from tictactoe import *
import numpy as np

np.random.seed(1234)
np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf, suppress=True)
TTT = TicTacToe()

############ PLAY TIC-TAC-TOE BEFORE AI IS TRAINED ############

print('\n\n\n PLAY TIC-TAC-TOE BEFORE AI IS TRAINED \n\n\n')
TTT.debug = True
TTT.play_the_whole_game(startplayer=1, humanplayer=1)
TTT.play_the_whole_game(startplayer=2, humanplayer=1)

############ TRAIN AI TO PLAY TIC-TAC-TOE ############

print('\n\n\n TRAIN AI TO PLAY TIC-TAC-TOE \n\n\n')
TTT.debug = False
## Choose policy iteration to train the AIs
# TTT.policy_training()
## Choose Q-Learning to train the AIs
TTT.QLearning_training()

############ PLAY TIC-TAC-TOE AFTER AI IS TRAINED ############

print('\n\n\n PLAY TIC-TAC-TOE AFTER AI IS TRAINED \n\n\n')
TTT.debug = True
for iter in range(3):
    TTT.play_the_whole_game(startplayer=1, humanplayer=1)
    TTT.play_the_whole_game(startplayer=2, humanplayer=1)
