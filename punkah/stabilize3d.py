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
from numpy import arange, arctan, array, cos, dot, sin, sqrt, tan, vstack
from numpy import average, std

from pymanoid.draw import draw_line, draw_polygon, draw_trajectory
from pymanoid.geometry import compute_polygon_hull
from pymanoid.optim import NonlinearProgram
from pymanoid.sim import gravity

from convex import ConvexProblem
from stabilize import Stabilizer


class ConvexProblem3D(ConvexProblem):

    def set_omega_f(self, omega_f):
        self.omega_f = omega_f

    def set_omega_i_lim(self, omega_i_min, omega_i_max):
        self.omega_i_max = omega_i_max
        self.omega_i_min = omega_i_min

    def set_z_bar(self, z_bar):
        self.z_bar = z_bar

    def set_zd_bar(self, zd_bar):
        self.zd_bar = zd_bar

    def compute_Phi(self):
        nlp = NonlinearProgram()
        g = -gravity[2]
        bc_integral_expr = 0.
        Phi_i = 0.  # Phi_0 = 0
        Phi_1 = None
        lambda_cost = 0.
        lambda_f = self.omega_f ** 2
        lambda_guess = lambda_f
        lambda_prev = lambda_f
        for i in xrange(self.N):
            Phi_lb = self.s_sq[i + 1] * self.lambda_min
            Phi_ub = self.s_sq[i + 1] * self.lambda_max
            if i == self.N - 1:
                if self.omega_i_min is not None:
                    Phi_lb = max(Phi_lb, self.omega_i_min ** 2)
                if self.omega_i_max is not None:
                    Phi_ub = min(Phi_ub, self.omega_i_max ** 2)
            Phi_next = nlp.new_variable(
                'Phi_%d' % (i + 1),  # from Phi_1 to Phi_N
                dim=1,
                init=[self.s_sq[i + 1] * lambda_guess],
                lb=[Phi_lb],
                ub=[Phi_ub])
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
        bc_cvx_obj = bc_integral_expr - (self.z_bar / g) * casadi_sqrt(Phi_N)
        nlp.add_equality_constraint(bc_cvx_obj, self.zd_bar / g)
        nlp.add_equality_constraint(Phi_1, self.Delta[0] * lambda_f)
        nlp.extend_cost(lambda_cost)
        nlp.create_solver()
        Phi_1_N = nlp.solve()
        Phi = array([0.] + list(Phi_1_N))  # preprend Phi_0 = 0
        if __debug__:
            self.solve_times.append(nlp.solve_time)
        self.optimal_found = nlp.optimal_found
        return Phi

    def plot_s(self):
        import pylab
        super(ConvexProblem3D, self).plot_s()
        pylab.step(
            [0., 1.], [self.omega_i_min ** 2] * 2, 'g', linestyle='--',
            where='post')
        pylab.step(
            [0., 1.], [self.omega_i_max ** 2] * 2, 'g', linestyle='--',
            where='post')
        pylab.step(
            [0., 1.], [self.omega_f ** 2] * 2, 'k', linestyle='--',
            where='post')

    def plot_t(self):
        import pylab
        super(ConvexProblem3D, self).plot_t()
        pylab.step(
            [0., 2. * self.switch_times[-1]], [self.omega_i_min ** 2] * 2, 'g',
            linestyle='--', where='post')
        pylab.step(
            [0., 2. * self.switch_times[-1]], [self.omega_i_max ** 2] * 2, 'g',
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
        g, Delta, Phi, N = -gravity[2], self.Delta, self.Phi, self.N
        out_bc_integral = sum(
            Delta[i] / (sqrt(Phi[i+1]) + sqrt(Phi[i])) for i in xrange(N))
        out_bc_cvx_obj = out_bc_integral - (self.z_bar / g) * sqrt(Phi[N])
        out_omega_f = sqrt(Phi[1] / Delta[0])
        print ""
        print "NLP perfs"
        print "---------"
        print "Limit state: %.1f%%" % succ(self.omega_f, out_omega_f)
        print "Boundedness: %.1f%%" % succ(self.zd_bar / g, out_bc_cvx_obj)
        print "Solve time:  %.1f +/- %.1f ms over %d samples" % (
            1000 * average(self.solve_times), 1000 * std(self.solve_times),
            len(self.solve_times))
        print ""
        self.succ = 100.

    def print_instance(self):
        print "Delta = %s;" % str(list(self.Delta))
        print "g = %f;" % -gravity[2]
        print "lambda_max = %f;" % self.lambda_max
        print "lambda_min = %f;" % self.lambda_min
        print "omega_i_max = %f;" % self.omega_i_max
        print "omega_i_min = %f;" % self.omega_i_min
        print "s = %s;" % str(list(self.s))
        print "z_bar = %f;" % self.z_bar
        print "z_f = %f;" % (-gravity[2] / self.omega_f ** 2)
        print "zd_bar = %f;" % self.zd_bar


class Stabilizer3D(Stabilizer):

    def __init__(self, pendulum, lambda_min, lambda_max, nb_steps, cop_gain):
        super(Stabilizer3D, self).__init__(pendulum)
        assert cop_gain > 1.05, "CoP gain has to be strictly > 1"
        self.cop_gain = cop_gain
        self.draw_polygon = False
        self.solver = ConvexProblem3D(lambda_min, lambda_max, nb_steps)

    def compute_surface_frame(self):
        contact = self.pendulum.contact
        dot_ez_n = dot(self.e_z, contact.n)
        phi = arctan(-dot(self.e_x, contact.n) / dot_ez_n)
        theta = arctan(-dot(self.e_y, contact.n) / dot_ez_n)
        t = cos(phi) * self.e_x + sin(phi) * self.e_z
        b = cos(theta) * self.e_y + sin(theta) * self.e_z

        if __debug__:
            self.__frames = [
                draw_line(contact.p, contact.p + .2 * self.e_x, color='r'),
                draw_line(contact.p, contact.p + .2 * self.e_y, color='g'),
                draw_line(contact.p, contact.p + .2 * self.e_z, color='b'),
                # draw_line(contact.p, contact.p + .2 * t, color='m'),
                # draw_line(contact.p, contact.p + .2 * b, color='k'),
                draw_line(contact.p, contact.p + .2 * contact.n, color='c')]

        self.R_surf = vstack([t, b, contact.n]).T
        self.t = t
        self.b = b
        self.phi = phi
        self.theta = theta

    def compute_omega_lim(self):
        W, H = self.pendulum.contact.shape
        b_h = dot(self.b, self.pendulum.contact.b) / cos(self.theta)
        b_w = dot(self.b, self.pendulum.contact.t) / cos(self.theta)
        t_h = dot(self.t, self.pendulum.contact.b) / cos(self.phi)
        t_w = dot(self.t, self.pendulum.contact.t) / cos(self.phi)
        A = array([
            [+t_w, +b_w],
            [-t_w, -b_w],
            [+t_h, +b_h],
            [-t_h, -b_h]])
        b = array([W, W, H, H])
        u = b / self.cop_gain - dot(A, self.state.p[:2])
        v = dot(A, self.state.pd[:2])
        # Property: u * omega_i >= v
        omega_i_min, omega_i_max = -1000., +1000.
        for i in xrange(A.shape[0]):
            if u[i] > 1e-3:
                omega_i_min = max(omega_i_min, v[i] / u[i])
            elif u[i] < 1e-3:
                omega_i_max = min(omega_i_max, v[i] / u[i])
            elif v[i] > 0:  # u[i] is almost 0., so v[i] must be negative
                raise RuntimeError("CoP polygon is singular, on the bad side")
        if __debug__ and self.draw_polygon:
            vertices = compute_polygon_hull(A, b)
            points = [
                self.pendulum.contact.p + _v[0] * self.e_x + _v[1] * self.e_y
                for _v in vertices]
            color = 'r' if omega_i_min > omega_i_max else 'g'
            self.__polygon = draw_polygon(points, [0., 0., 1.], '%s.-#' % color)
        if omega_i_min > omega_i_max:
            raise RuntimeError("No feasible CoP for this feedback gain")
        if omega_i_max > 500:
            omega_i_max = None
        if omega_i_min < -500:
            omega_i_min = None
        return (omega_i_min, omega_i_max)

    def compute_controls(self):
        """
        Compute pendulum controls for the current simulation step.

        Returns
        -------
        cop : (3,) array
            COP coordinates in the world frame.
        push : scalar
            Leg push :math:`\\lambda \\geq 0`.
        """
        self.compute_surface_frame()
        omega_i_min, omega_i_max = self.compute_omega_lim()
        x, y, z = self.state.p
        xd, yd, zd = self.state.pd
        self.solver.set_z_bar(z - tan(self.phi) * x - tan(self.theta) * y)
        self.solver.set_zd_bar(zd - tan(self.phi) * xd - tan(self.theta) * yd)
        self.solver.set_omega_i_lim(omega_i_min, omega_i_max)
        self.solver.set_omega_f(sqrt(-gravity[2] / self.pendulum.z_target))
        self.solver.solve()
        omega_i = self.solver.get_omega_i()
        r_i = self.cop_gain * (self.state.p + self.state.pd / omega_i)
        alpha_i = r_i[0] / cos(self.phi)
        beta_i = r_i[1] / cos(self.theta)
        cop = dot(self.R_surf, [alpha_i, beta_i, 0.])
        return cop, self.solver.get_lambda()

    def draw_solution(self):
        if self.solver is None or self.T is None:
            return []
        N, solver = self.solver.N, self.solver
        if solver.lambda_ is None:
            solver.compute_full_trajectory()
        virt_pendulum = self.pendulum.copy()
        virt_pendulum.hide()
        max_time = solver.switch_times[-1] * 1.1
        omega_i = solver.get_omega_i()
        k = self.cop_gain
        r_i = k * (self.state.p + self.state.pd / omega_i)
        alpha_i = r_i[0] / cos(self.phi)
        beta_i = r_i[1] / cos(self.theta)
        points = []
        for i in xrange(solver.N):
            t_i = solver.switch_times[i]
            t_next = solver.switch_times[i + 1] if i < N - 1 else max_time
            virt_pendulum.set_lambda(solver.lambda_[N - i - 1])
            dt = (t_next - t_i) / 10.
            for t in arange(t_i, t_next, dt):
                s = solver.s_from_t(t)
                omega_t = solver.omega_from_t(t)
                alpha = alpha_i * (s * omega_t / omega_i) ** (k - 1)
                beta = beta_i * (s * omega_t / omega_i) ** (k - 1)
                cop = dot(self.R_surf, [alpha, beta, 0.])
                virt_pendulum.set_cop(cop)
                virt_pendulum.integrate(dt)
            points.append(virt_pendulum.com.p)
        points.append(self.pendulum.target)
        return draw_trajectory(points, color='m')
