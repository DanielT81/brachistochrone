import numpy as np
from math import sin, cos, pi, sqrt, atan

from ipython_genutils.testing.decorators import skipif


def sqr(var):
    return var**2


    #
g = 9.81
def physics(start_vel, def_vec):
    if start_vel > 0:
        print('physics: the starting velocity doesnt make sense here')
        return
    deltaX = def_vec[0]
    deltaY = def_vec[1]
    deltaS = sqrt(sqr(deltaX) + sqr(deltaY))
    cos_phi = sqrt(1/(1+sqr(deltaX / deltaY)))
    a = ((-0.5) * g * cos_phi * (-deltaY / sqrt(deltaY**2))) #- (1/2 * 0.1 (air drag coefficient) * area of the object * velocity**2)
    b = start_vel
    c = deltaS
    t = np.roots([a,b,c])
    #if str(t[0])[len(t[0])] == 'j' or str(t[1])[len(t[1])] == 'j':
    if type(t[0]) == np.complex128 or type(t[1]) == np.complex128:
        print('physics: problem with the imaginary unity')
        return
    t = t[np.argsort(t)]
    if t[1] < 0:
        print('physics: both t values are negative')
    print(t)

    #print(type(t), '\n' * 5, t)
    return t


physics(-100,[10000,-50000])