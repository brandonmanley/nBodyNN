# -*- coding: ascii -*-
"""
Calculates the Pythagorean 3-body problem using different values for
the smoothing length in the n-body code.
"""
from __future__ import print_function
# import numpy
# import time

from amuse.community.brutus.interface import Brutus 
from amuse.units import nbody_system
from amuse.units.quantities import AdaptingVectorQuantity
from matplotlib import pyplot
from amuse.datamodel import Particles


def new_particles():
    particles = Particles(3)

    particles.mass = [20, 4., 5.] | nbody_system.mass
    particles.position = [
        [1, 3, 0],
        [-2, -1, 0],
        [10, -1, 0],
    ] | nbody_system.length

    particles.velocity = [0., 0., 0.] | nbody_system.speed
    particles.radius = 0 | nbody_system.length

    return particles


def run_pyth(interface, tend=100, dt=0.125, parameters=[]):

    code = interface()

    for name, value in parameters:
        setattr(code.parameters, name, value)

    code.particles.add_particles(new_particles())
    code.commit_particles()

    x = AdaptingVectorQuantity()
    y = AdaptingVectorQuantity()
    t = 0. | nbody_system.time
    while(t < tend - dt / 2):
        print("dt", dt)
        t = t + dt
        code.evolve_model(t)
        x.append(code.particles.x)
        y.append(code.particles.y)
    code.stop()

    return x, y


if __name__ == "__main__":
    codes_to_run = [('Brutus0, $\eta=0.03$', Brutus, [("dt_param", 0.0000003)]),
                    ('Brutus0, $\eta=0.01$', Brutus, [("dt_param", 0.01)])]
                    # ('Brutus0, $\eta=0.003$', Brutus, [("dt_param", 0.003)]),
                    # ('Brutus0, $\eta=0.001$', Brutus, [("dt_param", 0.001)])]
    N = (len(codes_to_run) - 1) / 2 + 1
    f = pyplot.figure(figsize=(8, 4 * N))

    for i, (label, interface, parameters) in enumerate(codes_to_run):
        x, y = run_pyth(interface, tend=100 | nbody_system.time,
                        dt=0.0625 | nbody_system.time, parameters=parameters)
        x = x.value_in(nbody_system.length)
        y = y.value_in(nbody_system.length)

        print(type(x))
        
        subplot = f.add_subplot(N, 2, i + 1)
        subplot.plot(x[:, 0], y[:, 0], 'r')
        subplot.plot(x[:, 1], y[:, 1], 'b')
        subplot.plot(x[:, 2], y[:, 2], 'g')
        subplot.set_title(label)
        subplot.set_xlim(-8, 8)
        subplot.set_ylim(-6, 6)
    pyplot.show()

'''
notes:

- Units:    variable | unit       is syntax for representing variables and units
            eg  mass=100 | kg      means 100 kg

            nbody_system includes 3 units: time, length, mass and sets G = 1

'''