import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as AA
#from ipython_genutils.testing.decorators import skipif


def sqr(var):
    return var**2


    #
def vector(minuend, subtrahend):
    result = AA(minuend) - AA(subtrahend)
    return result


    #
def new_point(start_point, end_point):
    def_point = AA(start_point) + 0.5*AA(vector(end_point, start_point))
    return def_point


    #
def norm_vec(def_vec):
    def_norm_vec = np.array([float(-def_vec[0]),float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1])))
    return def_norm_vec


    #
def sort_arr():
    iterations = 0
    total_iterations = np.size(new_arr, 0)
    while iterations <= total_iterations:
        if new_arr[iterations, 1] != 0.0:
            iterations +=1
        else:
            new_arr[:iterations] = new_arr[np.argsort(new_arr[:iterations, [1][0]])]
            arr_len = iterations
            break


    #
def optimizer(p1,p2):
    new_point = new_point(p2, p1)
    def_vec = vector(p2, p1)
    norm_vec = norm_vec(def_vec)

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
        return
    if t[0] > 0:
        print('physics: both t values are positive')
        print(t)
        return
    t_result = t[1]
    v_result = start_vel + a * cos_phi * t_result
    print([t_result, v_result])
    return [t_result, v_result]




print(new_point([0,10],[10,0]))
'''
physics(0,[10,-10])
physics(0,[-10,10])
'''