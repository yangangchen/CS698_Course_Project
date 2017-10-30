#####################################################
#   CS 698 Course Project: Robotic Control of Pendulum Swing-up
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
#   This code implements Robotic Control of Pendulum Swing-up. The example is from
# Kenji Doya, Reinforcement Learning in Continuous Time and Space
#   The methods implemented is Dynamic programming (policy iteration)
#####################################################

from HJB import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(1234)
np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf, suppress=True)

## Reinforcement learning
###############

H = HJB()

H.policy_iteration()

## Plot the optimal control and optimal value function
###############

myPlot3D(H.Theta, H.Omega, H.V, xlabel='\n$\\theta$', ylabel='\n$\omega$',
         zlabel='$V^a(s)$ \n', title='Optimal value function',
         set_num_xticks=5, set_num_yticks=5, set_num_zticks=3, filename='HJB-Value.pdf')
myPlot3D(H.Theta, H.Omega, H.C, xlabel='\n$\\theta$', ylabel='\n$\omega$',
         zlabel='$a^*(s)$ \n', title='Optimal action (control)',
         set_num_xticks=5, set_num_yticks=5, set_num_zticks=3, filename='HJB-Control.pdf')

## Animation
###############

T = 10
nT = 1000

# state_ini = np.array([0, 1])
# state_ini = np.array([math.pi, 1])
state_ini = np.array([math.pi, 0])

state = H.simulation(state_ini=state_ini, T=T, nT=nT)
action = H.func_action(state)
reward = H.func_reward(state, action)

x = np.sin(state[:, 0])
y = np.cos(state[:, 0])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
action_template = 'engine torque = %.1f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
action_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    action_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]
    line.set_data(thisx, thisy)  # Draw a line between [0, 0] and [x[i], y[i]]
    time_text.set_text(time_template % (i * T / nT))
    action_text.set_text(action_template % (action[i]))
    return line, time_text, action_text


ani = animation.FuncAnimation(fig, animate, np.arange(0, nT),
                              interval=25, blit=True, init_func=init)

plt.show()
