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

from casadi import sqrt as casadi_sqrt
from numpy import array, sqrt
from numpy import average, std

import pymanoid

from pymanoid.draw import draw_trajectory
from pymanoid.misc import error, info
from pymanoid.optim import NonlinearProgram
from pymanoid.sim import gravity
from pymanoid.transformations import apply_transform

from convex import ConvexProblem
from pendulum import InvertedPendulum
from stabilize import Stabilizer


class ConvexProblem2D(ConvexProblem):

    def set_bc_integral(self, bc_integral):
        self.bc_integral = bc_integral

    def set_omega_i(self, omega_i):
        self.omega_i = omega_i

    def set_omega_f(self, omega_f):
        self.omega_f = omega_f

    def compute_Phi(self):
        nlp = NonlinearProgram()
        bc_integral_expr = 0.
        Phi_i = 0.  # Phi_0 = 0
        Phi_1 = None
        lambda_cost = 0.
        lambda_guess = self.omega_i ** 2
        lambda_prev = self.omega_f ** 2
        for i in xrange(self.N):
            Phi_next = nlp.new_variable(
                'Phi_%d' % (i + 1),  # from Phi_1 to Phi_N
                dim=1,
                init=[self.s_sq[i + 1] * lambda_guess],
                lb=[self.s_sq[i + 1] * self.lambda_min],
                ub=[self.s_sq[i + 1] * self.lambda_max])
            if Phi_1 is None:
                Phi_1 = Phi_next
            bc_integral_expr += self.Delta[i] / (
                casadi_sqrt(Phi_next) + casadi_sqrt(Phi_i))
            lambda_i = (Phi_next - Phi_i) / self.Delta[i]
            lambda_cost += ((lambda_i - lambda_prev)) ** 2
            lambda_prev = lambda_i
            nlp.add_constraint(
                Phi_next - Phi_i,
                lb=[self.Delta[i] * self.lambda_min],
                ub=[self.Delta[i] * self.lambda_max])
            Phi_i = Phi_next
        Phi_N = Phi_next
        nlp.add_equality_constraint(bc_integral_expr, self.bc_integral)
        nlp.add_equality_constraint(Phi_1, self.Delta[0] * self.omega_f ** 2)
        nlp.add_equality_constraint(Phi_N, self.omega_i ** 2)
        nlp.extend_cost(lambda_cost)
        nlp.create_solver()
        Phi_1_N = nlp.solve()
        Phi = array([0.] + list(Phi_1_N))  # preprend Phi_0 = 0
        if __debug__:
            assert len(Phi) == self.N + 1
            self.solve_times.append(nlp.solve_time)
        self.optimal_found = nlp.optimal_found
        return Phi

    def plot_s(self):
        import pylab
        super(ConvexProblem2D, self).plot_s()
        pylab.step(
            [0., 1.], [self.omega_i ** 2] * 2, 'g', linestyle='--',
            where='post')
        pylab.step(
            [0., 1.], [self.omega_f ** 2] * 2, 'k', linestyle='--',
            where='post')

    def plot_t(self):
        import pylab
        super(ConvexProblem2D, self).plot_t()
        pylab.step(
            [0., 2. * self.switch_times[-1]], [self.omega_i ** 2] * 2, 'g',
            linestyle='--', where='post')
        pylab.step(
            [0., 2. * self.switch_times[-1]], [self.omega_f ** 2] * 2, 'k',
            linestyle='--', where='post')

    def print_debug_info(self):
        def succ(a, b):
            if abs(a) < 1e-10:
                return 100. if abs(b) < 1e-10 else 0.
            return 100 * (1. - abs(a - b) / abs(a))
        if self.Phi is None:
            return
        out_bc_integral = sum(self.Delta[i] / (
            sqrt(self.Phi[i + 1]) + sqrt(self.Phi[i]))
            for i in xrange(self.N))
        out_omega_i = sqrt(self.Phi[self.N])
        out_omega_f = sqrt(self.Phi[1] / self.Delta[0])
        succ_omega_i = succ(self.omega_i, out_omega_i)
        succ_omega_f = succ(self.omega_f, out_omega_f)
        succ_bc_integral = succ(self.bc_integral, out_bc_integral)
        print ""
        print "NLP perfs"
        print "---------"
        print "Init. state: %.1f%%" % succ_omega_i
        print "Limit state: %.1f%%" % succ_omega_f
        print "Boundedness: %.1f%%" % succ_bc_integral
        print "Solve time:  %.1f +/- %.1f ms over %d samples" % (
            1000 * average(self.solve_times), 1000 * std(self.solve_times),
            len(self.solve_times))
        print ""


class Stabilizer2D(Stabilizer):

    def __init__(self, pendulum, lambda_min, lambda_max, nb_steps):
        super(Stabilizer2D, self).__init__(pendulum)
        self.solver = ConvexProblem2D(lambda_min, lambda_max, nb_steps)

    def z_crit(self, state):
        """
        Zero-capturability indicator from [Koolen2016]_.

        Notes
        -----
        It is a necessary, non-sufficient indicator in our setting where
        :math:`0 < \\lambda_\\text{min} \\leq \\lambda \\leq
        \\lambda_\\text{max}`.

        References
        ----------
        .. [Koolen2016] "Balance control using center of mass height variation:
           Limitations imposed by unilateral contact", T. Koolen, M. Posa and R.
           Tedrake, IEEE-RAS International Conference on Humanoid Robots,
           November 2016.
        """
        z, zd = state.z, state.zd
        omega = -state.xd / state.x
        return z + zd / omega + 0.5 * gravity[2] / omega ** 2

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
        if __debug__:
            z_crit = self.z_crit(self.state)
            if z_crit < 0.:
                error("Cannot stabilize: z_crit=%.1f < 0" % z_crit)
            else:  # z_crit >= 0.
                info("Maybe capturable: z_crit=%.1f > 0" % z_crit)
        g = -gravity[2]
        omega_i = -self.state.xd / self.state.x
        bc_integral = (self.state.zd + omega_i * self.state.z) / g
        if abs(self.state.yd + omega_i * self.state.y) > 1e-10:
            raise Exception("2D balance only applies to planar motions")
        self.solver.set_bc_integral(bc_integral)
        self.solver.set_omega_i(omega_i)
        self.solver.set_omega_f(sqrt(g / self.pendulum.z_target))
        self.solver.solve()
        cop = array([0., 0., 0.])
        return cop, self.solver.get_lambda()

    def draw_solution(self):
        if self.solver is None or self.T is None:
            return []
        N, solver = self.solver.N, self.solver
        if solver.lambda_ is None:
            solver.compute_full_trajectory()
        origin = pymanoid.Contact(shape=(1e-3, 1e-3), pos=[0, 0, 0])
        p, pd = self.state.p, self.state.pd
        virt_pendulum = InvertedPendulum(1., p, pd, origin, None)
        virt_pendulum.hide()
        points = []
        max_time = solver.switch_times[-1] * 2
        for i in xrange(solver.N):
            t_i = solver.switch_times[i]
            t_next = solver.switch_times[i + 1] if i < N - 1 else max_time
            virt_pendulum.set_lambda(solver.lambda_[N - i - 1])
            virt_pendulum.integrate(t_next - t_i)
            points.append(apply_transform(self.T, virt_pendulum.com.p))
        return draw_trajectory(points, color='c')
