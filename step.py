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

import IPython
import os
import sys

from numpy import array, cross, dot
from time import sleep

try:  # use local pymanoid submodule
    script_path = os.path.realpath(__file__)
    sys.path = [os.path.dirname(script_path) + '/pymanoid'] + sys.path
    import pymanoid
except:  # this is to avoid warning E402 from Pylint :p
    pass

from pymanoid import Point, Stance
from pymanoid.draw import draw_line
from pymanoid.misc import info, norm, normalize, warn

from punkah import InvertedPendulum, Stabilizer3D


class SolutionDrawer(pymanoid.Process):

    def __init__(self, stabilizer):
        super(SolutionDrawer, self).__init__()
        self.handles = None
        self.stabilizer = stabilizer

    def on_tick(self, sim):
        self.handles = self.stabilizer.draw_solution()


class StateEditor(pymanoid.Process):

    def __init__(self, pendulum, vel_to_pos=0.4):
        super(StateEditor, self).__init__()
        init_end_pos = pendulum.com.p + pendulum.com.pd * vel_to_pos
        end_point = Point(init_end_pos, color='g', size=0.015)
        pendulum.pause_dynamics = True
        end_point.hide()  # TODO: remove
        self.line_handle = None
        self.end_point = end_point
        self.pendulum = pendulum
        self.vel_to_pos = vel_to_pos

    def hide(self):
        self.end_point.hide()
        if self.line_handle is not None:
            self.line_handle.Close()

    def pause(self):
        super(StateEditor, self).pause()
        self.pendulum.pause_dynamics = False
        self.hide()

    def resume(self):
        super(StateEditor, self).resume()
        pendulum.com.set_pos([0, 0, pendulum.z_target])
        robot.ik.solve()
        self.pendulum.pause_dynamics = True
        self.update_pendulum_velocity()
        self.end_point.show()

    def update_pendulum_velocity(self):
        cur_vel = (self.end_point.p - self.pendulum.com.p) / self.vel_to_pos
        self.pendulum.com.set_vel(cur_vel)

    def on_tick(self, sim):
        self.update_pendulum_velocity()
        self.line_handle = draw_line(
            self.pendulum.com.p, self.end_point.p, linewidth=3)


class WatcherProcess(pymanoid.Process):

    def __init__(self, pendulum):
        super(WatcherProcess, self).__init__()
        self.pendulum = pendulum

    def on_tick(self, sim):
        if norm(self.pendulum.com.p - self.pendulum.target) < 2e-3:
            info("Stopping simulation as pendulum converged")
            sim.stop()


def edit():
    state_editor.resume()
    if not sim.is_running:
        sim.start()


def go():
    state_editor.pause()
    if not sim.is_running:
        sim.start()


def plot_lambda():
    import pylab
    pylab.ion()
    pylab.clf()
    stabilizer.solver.plot_t()


def reset_camera(duration):
    sim.move_camera_to(array([
        [-0.29855084, 0.323731, -0.89781158,  2.06474113],
        [0.95328345,  0.14651351, -0.26416747,  0.70340455],
        [0.04602233, -0.93473634, -0.35234914,  1.52264643],
        [0.,  0.,  0.,  1.]]), duration=duration)


def record_video():
    """Function used to record the accompanying video."""
    sleep(2)
    sim.move_camera_to(array([
        [-0.99769796, -0.0164792, -0.06578162,  0.12574141],
        [0.05827919,  0.28763275, -0.95596597,  1.10682797],
        [0.0346745, -0.957599, -0.28601021,  1.14686227],
        [0.,  0.,  0.,  1.]]), duration=1.)
    sleep(2)
    reset_camera(duration=1.)
    sleep(2)
    go()
    while sim.is_running:
        sleep(1)
    sleep(2)
    edit()


def usage():
    print """
-------------------------------------------------------------------------------

USAGE
======

You can start by calling the following functions:

    edit()        -- Editor mode*
    go()          -- Execute motion from current state*
    plot_lambda() -- Show the MPC trajectory in lambda space
    sim.stop()    -- Stop the simulation altogether
    usage()       -- Show this message again

* BEWARE: once in editor/execution modes, IPOPT will print its (verbose)
  output in the terminal. I don't know yet how to make it quiet.

-------------------------------------------------------------------------------
"""


if __name__ == "__main__":
    sim = pymanoid.Simulation(dt=3e-2)
    try:  # use HRP4 if available
        robot = pymanoid.robots.HRP4()
    except:  # otherwise use default model
        robot = pymanoid.robots.JVRC1()
    robot.set_transparency(0.5)
    sim.set_viewer()
    sim.viewer.SetBkgndColor([1, 1, 1])
    reset_camera(duration=0)
    contact = pymanoid.Contact(
        shape=(0.12, 0.06),
        pos=[0.15, -0.15, 0.],
        # rpy=[-0.19798375, 0.13503151, 0],
        rpy=[-0.35, -0.35, 0.05],
        friction=0.7)
    mass = 38.  # [kg]
    z_target = 0.8  # [m]
    init_pos = array([0., 0., z_target])
    vel = (0.7, 0.1, 0.2)
    draw_parabola = True
    if "--comanoid" in sys.argv:
        try:
            import comanoid
            vel = comanoid.setup(sim, robot, contact)
            draw_parabola = False
        except ImportError:
            warn("comanoid module not available, switching to default")
    delta = init_pos - contact.p
    e_z = array([0., 0., 1.])
    e_x = -normalize(delta - dot(delta, e_z) * e_z)
    e_y = cross(e_z, e_x)
    init_vel = vel[0] * e_x + vel[1] * e_y + vel[2] * e_z
    pendulum = InvertedPendulum(mass, init_pos, init_vel, contact, z_target)
    pendulum.draw_parabola = draw_parabola
    stance = Stance(com=pendulum.com, right_foot=contact)
    stance.bind(robot)
    robot.ik.solve()

    g = -sim.gravity[2]
    lambda_min = 0.1 * g
    lambda_max = 2.0 * g
    stabilizer = Stabilizer3D(
        pendulum, lambda_min, lambda_max, nb_steps=10, cop_gain=2.)

    drawer = SolutionDrawer(stabilizer)
    state_editor = StateEditor(pendulum)

    sim.schedule(stabilizer)
    sim.schedule(pendulum)
    sim.schedule(robot.ik)
    sim.schedule_extra(drawer)
    sim.schedule_extra(state_editor)
    sim.schedule_extra(WatcherProcess(pendulum))
    sim.step()

    usage()

    if IPython.get_ipython() is None:
        IPython.embed()
