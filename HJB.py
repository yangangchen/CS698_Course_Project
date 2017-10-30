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

import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.interpolate import interp2d
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator


class HJB:
    ## Initialization of the class
    def __init__(self):
        self.rho = 1
        self.ntheta = 50
        self.nomega = 200
        self.N = (self.ntheta + 1) * (self.nomega + 1)
        self.dtheta = 2 * math.pi / self.ntheta
        self.domega = 20 / self.nomega
        self.gridtheta = np.linspace(-math.pi, math.pi, self.ntheta + 1)
        self.gridomega = np.linspace(-10, 10, self.nomega + 1)
        self.Omega, self.Theta = np.meshgrid(self.gridomega, self.gridtheta)
        self.vTheta = self.Theta.reshape(self.N)
        self.vOmega = self.Omega.reshape(self.N)

        self.C = np.zeros((self.ntheta + 1, self.nomega + 1))
        self.vC = self.C.reshape(self.N)
        self.V = np.zeros((self.ntheta + 1, self.nomega + 1))
        self.vV = self.V.reshape(self.N)

    ################# Dynamic programming (policy iteration) #################

    ## Dynamic programming (policy iteration): The step of policy evaluation
    def policy_evaluation(self):
        self.A = sps.lil_matrix((self.N, self.N))
        self.vF = np.zeros(self.N)

        mu1 = self.vOmega / self.dtheta
        mu2 = (- 0.01 * self.vOmega + 9.8 * np.sin(self.vTheta) + self.vC) / self.domega
        vF = np.cos(self.vTheta) - 0.05 * self.vC ** 2
        for i in range(self.ntheta + 1):
            for j in range(self.nomega + 1):
                I = (self.nomega + 1) * i + j
                if mu1[I] >= 0:
                    if i == self.ntheta:
                        self.A[I, I] = 1
                        self.A[I, (self.nomega + 1) * 0 + j] = -1
                        self.vF[I] = 0
                    else:
                        if mu2[I] >= 0:
                            self.A[I, I] = mu1[I] + mu2[I] + self.rho
                            self.A[I, (self.nomega + 1) * (i + 1) + j] = -mu1[I]
                            if j < self.nomega:
                                self.A[I, (self.nomega + 1) * i + (j + 1)] = -mu2[I]
                            else:  # NBC
                                self.A[I, (self.nomega + 1) * i + (j - 1)] = -mu2[I]
                        elif mu2[I] < 0:
                            self.A[I, I] = mu1[I] - mu2[I] + self.rho
                            self.A[I, (self.nomega + 1) * (i + 1) + j] = -mu1[I]
                            if j > 0:
                                self.A[I, (self.nomega + 1) * i + (j - 1)] = mu2[I]
                            else:  # NBC
                                self.A[I, (self.nomega + 1) * i + (j + 1)] = mu2[I]
                        self.vF[I] = vF[I]
                elif mu1[I] < 0:
                    if i == 0:
                        self.A[I, I] = 1
                        self.A[I, (self.nomega + 1) * self.ntheta + j] = -1
                        self.vF[I] = 0
                    else:
                        if mu2[I] >= 0:
                            self.A[I, I] = -mu1[I] + mu2[I] + self.rho
                            self.A[I, (self.nomega + 1) * (i - 1) + j] = mu1[I]
                            if j < self.nomega:
                                self.A[I, (self.nomega + 1) * i + (j + 1)] = -mu2[I]
                            else:  # NBC
                                self.A[I, (self.nomega + 1) * i + (j - 1)] = -mu2[I]
                        elif mu2[I] < 0:
                            self.A[I, I] = -mu1[I] - mu2[I] + self.rho
                            self.A[I, (self.nomega + 1) * (i - 1) + j] = mu1[I]
                            if j > 0:
                                self.A[I, (self.nomega + 1) * i + (j - 1)] = mu2[I]
                            else:  # NBC
                                self.A[I, (self.nomega + 1) * i + (j + 1)] = mu2[I]
                        self.vF[I] = vF[I]
        self.vR = self.vF - self.A.dot(self.vV)
        self.nR = np.sum(abs(self.vR))
        self.vV[:] = spsl.spsolve(self.A.tocsr(), self.vF)

    ## Dynamic programming (policy iteration): The step of policy improvement
    def policy_improvement(self):
        vC0 = 0.01 * self.vOmega - 9.8 * np.sin(self.vTheta)
        dw_V_forward = np.zeros((self.ntheta + 1, self.nomega + 1))
        dw_V_forward[:, :-1] = (self.V[:, 1:] - self.V[:, :-1]) / self.domega
        dw_V_forward[:, -1] = -dw_V_forward[:, -2]
        dw_vV_forward = dw_V_forward.reshape(self.N)
        dw_V_backward = np.zeros((self.ntheta + 1, self.nomega + 1))
        dw_V_backward[:, 1:] = (self.V[:, 1:] - self.V[:, :-1]) / self.domega
        dw_V_backward[:, 0] = -dw_V_backward[:, 1]
        dw_vV_backward = dw_V_backward.reshape(self.N)

        type1 = (vC0 <= -5)  # mu2 >= 0 for all C in [-5,5]
        self.vC[type1] = np.vstack([
            np.vstack([
                10 * dw_vV_forward[type1],
                5 * np.ones(sum(type1))]).min(axis=0),
            -5 * np.ones(sum(type1))]).max(axis=0)

        type2 = (vC0 > -5) & (vC0 < 5)
        vCstar1 = np.vstack([
            np.vstack([
                10 * dw_vV_backward,
                vC0]).min(axis=0),
            -5 * np.ones(self.N)]).max(axis=0)
        vCstar2 = np.vstack([
            np.vstack([
                10 * dw_vV_forward,
                5 * np.ones(self.N)]).min(axis=0),
            vC0]).max(axis=0)
        mu2star1 = - 0.01 * self.vOmega + 9.8 * np.sin(self.vTheta) + vCstar1
        vVstar1 = mu2star1 * dw_vV_backward - 0.05 * vCstar1 ** 2
        mu2star2 = - 0.01 * self.vOmega + 9.8 * np.sin(self.vTheta) + vCstar2
        vVstar2 = mu2star2 * dw_vV_forward - 0.05 * vCstar2 ** 2
        self.vC[type2 & (vVstar1 >= vVstar2)] = vCstar1[type2 & (vVstar1 >= vVstar2)]
        self.vC[type2 & (vVstar2 > vVstar1)] = vCstar2[type2 & (vVstar2 > vVstar1)]

        type3 = (vC0 >= 5)  # mu2 <= 0 for all C in [-5,5]
        self.vC[type3] = np.vstack([
            np.vstack([
                10 * dw_vV_backward[type3],
                5 * np.ones(sum(type3))]).min(axis=0),
            -5 * np.ones(sum(type3))]).max(axis=0)

    ## Dynamic programming (policy iteration): Complete process
    def policy_iteration(self, max_iter=100):
        for iter in range(max_iter):
            self.policy_evaluation()
            self.policy_improvement()
            print(self.nR)
            if self.nR <= 1e-6:
                print('Policy iteration succeeds in ' + str(iter) + ' iterations!')
                break

    ## Simulation
    def func_reward(self, state, action):
        return np.cos(state[:, 0]) - 0.05 * action ** 2

    def func_action(self, state):
        faction = interp2d(self.gridtheta, self.gridomega, self.C.transpose(), kind='linear')
        return np.array([faction(state[i, 0], state[i, 1])[0] for i in range(len(state))])

    def func_RHS(self, y, t):
        faction = interp2d(self.gridtheta, self.gridomega, self.C.transpose(), kind='linear')
        theta, omega = y  # unpack current values of y
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        derivs = [omega,
                  - 0.01 * omega + 9.8 * math.sin(theta)
                  + faction(theta, omega)]
        return derivs

    def simulation(self, state_ini, T, nT):
        return odeint(self.func_RHS, state_ini, np.linspace(0, T, nT + 1))


################################################

def myPlot3D(X, Y, Z, title=None, xlabel=None, ylabel=None, zlabel=None,
             xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
             show_colorbar=False, set_num_xticks=None, set_num_yticks=None,
             set_num_zticks=None, filename=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
    ax.pbaspect = [1.0, 1.0, 0.1]
    if title is not None:
        ax.set_title(title, fontsize=28)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=28)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=28)
    if zlabel is not None:
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zlabel, fontsize=28, rotation=0)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if zmin is not None and zmax is not None:
        ax.set_zlim(zmin, zmax)
    if show_colorbar:
        fig.colorbar(surf, shrink=0.7)
    if set_num_xticks is not None:
        ax.xaxis.set_major_locator(LinearLocator(set_num_xticks))
    if set_num_yticks is not None:
        ax.yaxis.set_major_locator(LinearLocator(set_num_yticks))
    if set_num_zticks is not None:
        ax.zaxis.set_major_locator(LinearLocator(set_num_zticks))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
