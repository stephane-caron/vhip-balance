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

import os
import IPython
import sys

from numpy import array, average, dot, sqrt, std

import pymanoid

from pymanoid import Stance
from pymanoid.draw import draw_line
from pymanoid.misc import error, norm, normalize, warn

try:
    script_path = os.path.realpath(__file__)
    sys.path = [os.path.dirname(script_path) + '/..'] + sys.path
    from punkah import InvertedPendulum, Stabilizer2D, Stabilizer3D
except:  # this is to avoid warning E402 from Pylint
    pass


def sample_scenario(contact_dist=0.3, contact_rpy=0.3, z_crit_var=0.1):
    from numpy import random
    contact_pos = random.random(3) * [contact_dist, contact_dist, 0.]
    contact_rpy = random.random(3) * [contact_rpy, contact_rpy, contact_rpy]
    z_crit_des = 0.5 * pendulum.z_target + random.random() * z_crit_var
    zd = 0.5 * random.random()
    scenario = (contact_pos, contact_rpy, z_crit_des, zd)
    return scenario


def reset_state(scenario):
    contact_pos, contact_rpy, z_crit_des, zd = scenario
    contact.set_pos(contact_pos)
    contact.set_rpy(contact_rpy)
    pendulum.com.set_pos([0, 0, pendulum.z_target])
    robot.ik.solve(max_it=100, impr_stop=1e-4)
    e_z = array([0., 0., 1.])
    e_x = pendulum.target - pendulum.com.p
    e_x = normalize(e_x - dot(e_x, e_z) * e_z)
    z_diff = (pendulum.com.z - contact.z) - z_crit_des
    x = -dot(e_x, (contact.p - pendulum.com.p))
    xd = - x / (2 * z_diff) * (-zd + sqrt(zd ** 2 + 2 * 9.81 * z_diff))
    vel = xd * e_x + zd * e_z
    if '--gui' in sys.argv:
        global vel_handle
        vel_handle = draw_line(pendulum.com.p, pendulum.com.p + 0.5 * vel)
    pendulum.com.set_vel(vel)
    sim.step()


if __name__ == "__main__":
    sim = pymanoid.Simulation(dt=3e-2)
    try:  # use HRP4 if available
        robot = pymanoid.robots.HRP4()
    except:  # otherwise use default model
        robot = pymanoid.robots.JVRC1()
    robot.set_transparency(0.5)
    robot.suntan()
    robot.add_shoulder_abduction_task(weight=1e-4)
    robot.add_shoulder_flexion_task(weight=1e-4)
    if '--gui' in sys.argv:
        sim.set_viewer()
        sim.viewer.SetBkgndColor([1, 1, 1])
        sim.viewer.SetCamera([
            [-0.43496840,  0.32393275, -0.84016074,  2.07345104],
            [+0.88804317,  0.30865173, -0.34075422,  0.77617097],
            [+0.14893562, -0.89431632, -0.42192002,  1.63681912],
            [+0.,  0.,  0.,  1.]])
    contact = pymanoid.Contact(
        shape=(0.12, 0.06),
        pos=[0.15, -0.15, 0.],
        # rpy=[-0.19798375, 0.13503151, 0],
        rpy=[-0.35, -0.35, 0.05],
        friction=0.7)
    mass = 38.  # [kg]
    z_target = 0.8  # [m]
    init_pos = array([0., 0., z_target])
    init_vel = 4. * (contact.p - init_pos) * array([1., 1., 0.])
    pendulum = InvertedPendulum(mass, init_pos, init_vel, contact, z_target)
    stance = Stance(com=pendulum.com, right_foot=contact)
    stance.bind(robot)
    robot.ik.solve()

    g = -sim.gravity[2]
    lambda_min = 0.1 * g
    lambda_max = 2.0 * g
    stabilizer_2d = Stabilizer2D(
        pendulum, lambda_min, lambda_max, nb_steps=10)
    stabilizer_3d = Stabilizer3D(
        pendulum, lambda_min, lambda_max, nb_steps=10, cop_gain=2.)

    sim.schedule(stabilizer_2d)
    sim.schedule(stabilizer_3d)
    sim.schedule(pendulum)
    sim.schedule(robot.ik)
    sim.step()

    nb_launches, nb_samples = 0, 0
    while nb_launches < 100 or nb_samples < 10000:
        scenario = sample_scenario()
        for stabilizer in [stabilizer_2d, stabilizer_3d]:
            stabilizer_2d.pause()
            stabilizer_3d.pause()
            stabilizer.resume()
            reset_state(scenario)
            if not stabilizer.solver.optimal_found:
                warn("Unfeasible sample")
                continue
            while norm(pendulum.com.p - pendulum.target) > 2e-3:
                if not stabilizer.solver.optimal_found or stabilizer.T is None:
                    error("Unfeasible state encountered during execution")
                    break
                sim.step()
                print "\n------------------------------------------\n"
                print "Launch %d for %s" % (
                    nb_launches + 1, type(stabilizer).__name__)
                stabilizer.solver.print_debug_info()
        nb_2d_samples = len(stabilizer_2d.solver.solve_times)
        nb_3d_samples = len(stabilizer_3d.solver.solve_times)
        nb_samples = min(nb_2d_samples, nb_3d_samples)
        nb_launches += 1

    print "Results"
    print "-------"
    for stabilizer in [stabilizer_2d, stabilizer_3d]:
        msg = "Solve time:  %.1f +/- %.1f ms over %d samples, %d launches" % (
            1000 * average(stabilizer.solver.solve_times),
            1000 * std(stabilizer.solver.solve_times),
            len(stabilizer.solver.solve_times), nb_launches)
        name = type(stabilizer).__name__
        print "%s: %s" % (name, msg)
    print ""

    if IPython.get_ipython() is None:
        IPython.embed()
