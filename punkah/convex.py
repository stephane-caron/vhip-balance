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

from bisect import bisect_right
from numpy import array, linspace, log, sqrt
from numpy import cosh, sinh, tanh


class ConvexProblem(object):

    """
    Find bounded solutions under feasibility constraints.

    Parameters
    ----------
    lambda_min : scalar
        Minimum leg stiffness (positive).
    lambda_max : scalar
        Maximum leg stiffness (positive).
    nb_steps : integer
        Number of phases where :math:`\\lambda(t)` is piecewise constant.
    """

    def __init__(self, lambda_min, lambda_max, nb_steps):
        s_list = [i * 1. / nb_steps for i in xrange(nb_steps)] + [1.]
        s_sq = [s ** 2 for s in s_list]
        Delta = [s_sq[i + 1] - s_sq[i] for i in xrange(nb_steps)]
        self.Delta = Delta
        self.N = nb_steps
        self.Phi = None
        self.lambda_ = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.s = s_list
        self.s_sq = s_sq
        self.solve_times = []
        self.switch_times = None

    def solve(self):
        self.Phi = None
        self.lambda_ = None
        self.switch_times = None
        self.Phi = self.compute_Phi()

    def get_lambda(self):
        assert self.Phi is not None
        Delta, Phi, N = self.Delta, self.Phi, self.N
        lambda_ = (Phi[N] - Phi[N - 1]) / Delta[N - 1]
        return lambda_

    def get_omega_i(self):
        return sqrt(self.Phi[self.N])

    def compute_lambda(self):
        lambda_ = [
            (self.Phi[i + 1] - self.Phi[i]) / self.Delta[i]
            for i in xrange(self.N)]
        lambda_ = array(lambda_ + [lambda_[-1]])
        self.lambda_ = lambda_

    def lambda_from_s(self, s):
        """
        Compute the leg stiffness :math:`\\lambda(s)` for a given path index.

        Parameters
        ----------
        s : scalar
            Path index between 0 and 1.

        Returns
        -------
        lambda_ : scalar
            Leg stiffness :math:`\\lambda(s)`.

        Note
        ----
        This function is not called by the stabilizer, but useful for debugging.
        """
        i = bisect_right(self.s, s) - 1 if s > 0 else 0
        assert self.s[i] <= s and (i == self.N or s < self.s[i + 1])
        return self.lambda_[i]

    def omega_from_s(self, s):
        """
        Compute :math:`\\omega(s)` for a given path index.

        Parameters
        ----------
        s : scalar
            Path index between 0 and 1.

        Returns
        -------
        omega : scalar
            Value of :math:`\\omega(s)`.

        Note
        ----
        This function is not called by the stabilizer, but useful for debugging.
        """
        if s < 1e-3:
            return sqrt(self.lambda_[0])
        i = bisect_right(self.s, s) - 1 if s > 0 else 0
        assert self.s[i] <= s and (i == self.N or s < self.s[i + 1])
        # integral from 0 to s of f(u) = 2 * u * lambda(u)
        f_integral = self.Phi[i] + self.lambda_[i] * (s ** 2 - self.s[i] ** 2)
        return sqrt(f_integral) / s

    def compute_switch_times(self):
        """
        Compute the times :math:`t_i` where :math:`s(t_i) = s_i`.

        Note
        ----
        Timing information is not used by the stabilizer, but useful for
        debugging. It enables the computation of `s(t)`,
        :math:`\\lambda(t)` and :math:`\\omega(t)`.
        """
        switch_times = [0.]
        switch_time = 0.
        for i in xrange(self.N - 1, 0, -1):
            num = sqrt(self.Phi[i + 1]) + sqrt(self.lambda_[i]) * self.s[i + 1]
            denom = sqrt(self.Phi[i]) + sqrt(self.lambda_[i]) * self.s[i]
            duration = log(num / denom) / sqrt(self.lambda_[i])
            switch_time += duration
            switch_times.append(switch_time)
        self.switch_times = switch_times

    def compute_full_trajectory(self):
        self.compute_lambda()
        self.compute_switch_times()

    def find_switch_time_before(self, t):
        """
        Find a switch time :math:`t_i` such that :math:`t_i \leq t < t_{i+1}`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        i : integer
            Switch-time index between 0 and N - 1.
        t_i : scalar
            Switch time in [s].
        """
        i = bisect_right(self.switch_times, t) - 1 if t > 0 else 0
        assert self.switch_times[i] <= t
        assert i == len(self.switch_times) - 1 or t < self.switch_times[i + 1]
        return i, self.switch_times[i]

    def s_from_t(self, t):
        """
        Compute the path index corresponding to a given time.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        s : scalar
            Path index `s(t)`.
        """
        i, t_i = self.find_switch_time_before(t)
        s_start = self.s[self.N - i]
        omega_ = sqrt(self.Phi[self.N - i]) / self.s[self.N - i]
        lambda_ = self.lambda_[self.N - i - 1]
        sqrt_lambda = sqrt(lambda_)
        x = sqrt_lambda * (t - t_i)
        return s_start * (cosh(x) - omega_ / sqrt_lambda * sinh(x))

    def lambda_from_t(self, t):
        """
        Compute the leg stiffness :math:`\\lambda(t)` to apply at time `t`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        lambda_ : scalar
            Leg stiffness :math:`\\lambda(t)`.
        """
        return self.lambda_from_s(self.s_from_t(t))

    def omega_from_t(self, t):
        """
        Compute the value of :math:`\\omega(t)`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        omega : scalar
            Value of :math:`\\omega(t)`.
        """
        i, t_i = self.find_switch_time_before(t)
        omega_ = sqrt(self.Phi[self.N - i]) / self.s[self.N - i]
        lambda_ = self.lambda_[self.N - i - 1]
        sqrt_lambda = sqrt(lambda_)
        x = sqrt_lambda * (t - t_i)
        z = sqrt_lambda / omega_
        return sqrt_lambda * (1. - z * tanh(x)) / (z - tanh(x))

    def plot_s(self):
        """
        Plot :math:`\\lambda(s)` and :math:`\\omega(s)^2` curves.
        """
        from pylab import grid, legend, plot, step, xlabel, ylim

        def subsample(s):
            s2 = [(s[i] + s[i + 1]) / 2. for i in xrange(len(s) - 1)]
            s2.extend(s)
            s2.sort()
            return s2

        s_more = subsample(subsample(self.s))
        omega_ = [self.omega_from_s(s) ** 2 for s in s_more]
        step(self.s, self.lambda_, 'b-', where='post', marker='o')
        plot(s_more, omega_, 'r-', marker='o')
        legend(('$\\lambda(s)$', '$\\omega(s)^2$'), loc='upper left')
        xlabel('$s$', fontsize=18)
        ymin = 0.9 * min(min(self.lambda_), min(omega_))
        ymax = 1.1 * max(max(self.lambda_), max(omega_))
        ylim((ymin, ymax))
        grid()

    def plot_t(self):
        """
        Plot :math:`\\lambda(t)` and :math:`\\omega(t)^2` curves.
        """
        from pylab import grid, legend, plot, step, xlabel, ylim
        times = list(self.switch_times) + [2 * self.switch_times[-1]]
        lambda_ = list(self.lambda_[-1::-1])
        trange = linspace(0, max(times), 20)
        omega_ = [self.omega_from_t(t) ** 2 for t in trange]
        step(times, lambda_, marker='o', where='pre')
        plot(trange, omega_, 'r-', marker='o')
        legend(('$\\lambda(t)$', '$\\omega(t)^2$'))
        xlabel('$t$', fontsize=18)
        ymin = 0.9 * min(min(self.lambda_), min(omega_))
        ymax = 1.1 * max(max(self.lambda_), max(omega_))
        ylim((ymin, ymax))
        grid()
