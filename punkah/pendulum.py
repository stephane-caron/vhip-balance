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

import pymanoid

from numpy import array, cosh, dot, sinh, sqrt

from pymanoid import PointMass
from pymanoid.draw import draw_line, draw_point, draw_trajectory
from pymanoid.misc import warn
from pymanoid.sim import gravity


class InvertedPendulum(pymanoid.Process):

    """
    Inverted pendulum model.

    Parameters
    ----------
    mass : scalar
        Total robot mass.
    pos : (3,) array
        Initial position in the world frame.
    vel : (3,) array
        Initial velocity in the world frame.
    contact : pymanoid.Contact
        Foot contact specification.
    z_target : scalar
        Desired CoM height above contact.
    """

    def __init__(self, mass, pos, vel, contact, z_target):
        super(InvertedPendulum, self).__init__()
        com = PointMass(pos, mass, vel)
        self.com = com
        self.contact = contact
        self.cop = array([0., 0., 0.])
        self.draw_parabola = False
        self.handles = {}
        self.hidden = False
        self.lambda_ = 9.81 * (com.z - contact.z)
        self.pause_dynamics = False
        self.z_target = z_target

    @property
    def target(self):
        return self.contact.p + array([0., 0., self.z_target])

    def copy(self):
        return InvertedPendulum(
            self.com.mass, self.com.p, self.com.pd, self.contact,
            self.z_target)

    def hide(self):
        self.com.hide()
        for gh in self.handles:
            gh.Close()
        self.hidden = True

    def set_cop(self, cop):
        """
        Update the CoP location on the contact surface.

        Parameters
        ----------
        cop : (3,) array
            CoP location in the *local* inertial frame, with origin at the
            contact point and axes parallel to the world frame.
        """
        if __debug__:
            cop_check = dot(self.contact.R.T, cop)
            if abs(cop_check[0]) > 1.05 * self.contact.shape[0] \
                    or abs(cop_check[1]) > 1.05 * self.contact.shape[1]:
                warn("CoP outside of contact area")
        self.cop = cop

    def set_lambda(self, lambda_):
        """
        Update the leg stiffness coefficient.

        Parameters
        ----------
        lambda_ : scalar
            Leg stiffness coefficient (positive).
        """
        self.lambda_ = lambda_

    def draw(self, step=0.02):
        self.handles['leg'] = draw_line(
            self.com.p, self.contact.p + self.cop, linewidth=4, color='g')
        self.handles['target'] = draw_line(self.contact.p, self.target, 'b')
        self.handles['target_p'] = draw_point(self.target, 'b', pointsize=0.01)
        if not __debug__ or not self.draw_parabola:
            return
        p, t = self.com.p, 0.
        parabola = []
        while p[2] > self.com.z - 1:
            p = self.com.p + self.com.pd * t + gravity * t ** 2 / 2
            parabola.append(p)
            t += step
        if not parabola:
            return
        self.handles['parabola'] = draw_trajectory(parabola, pointsize=0)

    def integrate(self, duration):
        omega = sqrt(self.lambda_)
        p0 = self.com.p
        pd0 = self.com.pd
        ch, sh = cosh(omega * duration), sinh(omega * duration)
        vrp = self.contact.p + self.cop - gravity / self.lambda_
        p = p0 * ch + pd0 * sh / omega - vrp * (ch - 1.)
        pd = pd0 * ch + omega * (p0 - vrp) * sh
        self.com.set_pos(p)
        self.com.set_vel(pd)

    def on_tick(self, sim):
        if not self.pause_dynamics:
            self.integrate(sim.dt)
        if not self.hidden:
            self.draw()
