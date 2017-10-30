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

from gridworld import *
import numpy as np

np.random.seed(1234)
np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf, suppress=True)

## Produce the same result as the website http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# GW = GridWorld(deadend=False)
## Produce another type of result
GW = GridWorld()

## Choose the reinforcement learning methods
# GW.policy_iteration()
# GW.MonteCarlo()
# GW.TD_Sarsa()
GW.TD_QLearning()
# GW.TDlambda_Sarsa(Lambda=0.9)
# GW.TDlambda_QLearning(Lambda=0.9)

## Show the results
GW.show_value()
GW.show_policy()
