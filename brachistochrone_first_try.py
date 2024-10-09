import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as AA
import scipy as sp
import time

start_time = time.process_time() #to track the computation-time
for i in range(1000000):
    pass

g = -9.81
index_number = 2
AIP = 4 #amount of iterations to create new points
ATP = 2**AIP + 1 #amount of total points in the system
arr_len = 2 #length of the non-zero values


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
def sort_arr():
    iterations = 0
    total_iterations = np.size(arr, 0)
    while iterations <= total_iterations:
        if arr[iterations, 1] != 0.0:
            iterations +=1
        else:
            arr[:iterations] = arr[np.argsort(arr[:iterations, [1][0]])]
            arr_len = iterations
            break


    #
def norm_vec(def_vec):
    def_norm_vec = np.array([float(-def_vec[0]),float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1])))
    return def_norm_vec


    #
def optimizer(p1,p2):
    new_point = new_point(p2, p1)
    def_vec = vector(p2, p1)
    norm_vec = norm_vec(def_vec)
    norm_vec_fac = 0
    start_vel = arrT[7]
    arrT[0] = [p1, p2, new_point, norm_vec, norm_vec_fac, physics()]]
    
    
    #
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


    #
def vec_arr():
    iterations = 0




arr = np.zeros([ATP,2]) #array with all the points and given indices to track manually
arrT = np.zeros([3,8], dtype=object) #read the README to get the structure

arr[0] = [0, 10] #setting the boundary conditions
arr[1] = [10, 0] #setting the boundary conditions
arrT[0] = [arr[0],      arr[1],      new_point(arr[0], arr[1]),      norm_vec(vector(arr[0],arr[1])),         0,       physics(arr[0], arr[1])[0], physics(arr[0], arr[1])[1],   new_point(arr[0],arr[1])] #setting the boundary condition so there is no need for an annoying if clause
#     starting point  end point      new point between the two       normal vector to the vector        norm_vec_factor       time_taken                end_velocity   newly created point 'new_point + norm_vec * norm_vec_factor'
#print(physics(arrT[0,6] + 1, vector(arrT[0,0], arrT[0,1])))
print(physics(-0, [10, -10]))
#print(vector(arrT[0,0], arrT[0,1]))





end_time = time.process_time()
#print('\n' * 5, f"Elapsed time: {end_time - start_time} seconds")