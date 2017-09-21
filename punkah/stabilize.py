#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of 3d-balance
# <https://github.com/stephane-caron/3d-balance>.
#
# 3d-balance is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# 3d-balance is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# 3d-balance. If not, see <http://www.gnu.org/licenses/>.

from numpy import array, cross, dot, vstack

import pymanoid

from pymanoid.misc import error, normalize, warn
from pymanoid.transformations import transform_from_R_p


class Stabilizer(pymanoid.Process):

    """
    Model predictive controller for 3D balance.

    Parameters
    ----------
    pendulum : InvertedPendulum
        Pendulum instance.
    """

    def __init__(self, pendulum):
        super(Stabilizer, self).__init__()
        state = pymanoid.Point([0., 0., 0.])
        state.hide()
        self.T = None
        self.e_x = None
        self.e_y = None
        self.e_z = None
        self.pendulum = pendulum
        self.solver = None  # ConvexProblem, instantiated by child class
        self.state = state

    def update_state(self):
        """
        Compute the stationary frame and current pendulum state in that frame.
        """
        com = self.pendulum.com
        contact = self.pendulum.contact
        delta = com.p - contact.p
        e_z = array([0., 0., 1.])
        e_x = -normalize(delta - dot(delta, e_z) * e_z)
        e_y = cross(e_z, e_x)
        R = vstack([e_x, e_y, e_z])  # from world to local frame
        p, pd = dot(R, delta), dot(R, com.pd)  # in local frame
        T = transform_from_R_p(R.T, contact.p)  # local to world frame
        self.T = T
        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z
        self.state.set_pos(p)
        self.state.set_vel(pd)

    def compute_controls(self):
        """
        Compute pendulum controls for the current simulation step.

        Returns
        -------
        cop : (3,) array
            COP coordinates in the world frame.
        lambda_ : scalar
            Leg stiffness coefficient :math:`\\lambda \\geq 0`.
        """
        raise NotImplementedError("to be implemented by child class")

    def on_tick(self, sim):
        """
        Update pendulum controls based on stabilizing solution.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.
        """
        self.update_state()
        try:
            cop, lambda_ = self.compute_controls()
            self.pendulum.set_cop(cop)
            self.pendulum.set_lambda(lambda_)
        except RuntimeError as e:
            if "CoP" in str(e) or "Ill-posed problem detected" in str(e):
                self.T = None  # don't try to draw trajectory
                error("Convex problem has no solution")
                warn("Details: %s" % str(e))
            else:  # surprise!
                raise
