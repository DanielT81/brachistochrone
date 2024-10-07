import numpy as np
from math import sin, cos, pi, sqrt, atan

from ipython_genutils.testing.decorators import skipif


def sqr(var):
    return var**2


    #
g = 9.81
def physics(start_vel, def_vec):
    deltaX = def_vec[0]
    deltaY = def_vec[1]
    deltaS = sqrt(sqr(deltaX) + sqr(deltaY))
    cos_phi = sqrt(1/(1+sqr(deltaX / deltaY)))
    a = (-0.5) * g * cos_phi
    b = start_vel
    c = deltaS
    t = np.roots([a,b,c])
    if type(t[0]) == 'numpy.complex128' or type(t[1]) == 'numpy.complex128':
        print('problem with the imaginary unity')
    #v1 = start_vel + (g * cos_phi) * t1
    #v2 = start_vel + (g * cos_phi) * t2
    #print((-0.5)*g*cos_phi)
    print(t)
    return t


physics(-4,[880,45])