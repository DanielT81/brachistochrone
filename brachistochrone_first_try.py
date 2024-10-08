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
    total_iterations = np.size(new_arr, 0)
    while iterations <= total_iterations:
        if new_arr[iterations, 1] != 0.0:
            iterations +=1
        else:
            new_arr[:iterations] = new_arr[np.argsort(new_arr[:iterations, [1][0]])]
            arr_len = iterations
            break


    #
def norm_vec(def_vec):
    def_norm_vec = np.array([float(-def_vec[0]),float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1])))
    return def_norm_vec


    #
def physics(start_vel, def_vec):
    deltaX = def_vec[0]
    deltaY = def_vec[1]
    deltaS = sqrt(sqr(deltaX) + sqr(deltaY))
    cos_phi = sqrt(1/(1+sqr(deltaX / deltaY)))
    a = (-0.5) * g * cos_phi
    b = start_vel
    c = deltaS
    try:
        t = np.roots()
        #t1 = (-start_vel + sqrt(sqr(start_vel) + (2 * g * cos_phi * deltaS))) / (g * cos_phi)
        #t2 = (-start_vel - sqrt(sqr(start_vel) + (2 * g * cos_phi * deltaS))) / (g * cos_phi)
        v1 = start_vel + (g * cos_phi) * t1
        v2 = start_vel + (g * cos_phi) * t2
        print((-0.5)*g*cos_phi)
        return AA([t1, t2, v1, v2])
    except ValueError:
        1==1
        print('problem with the imaginary unity')


    #
def vec_arr():
    iterations = 0




arr = np.zeros([ATP,2]) #array with all the points and given indices to track manually
arrT = np.zeros([2,8], dtype=object) #read the README to get the structure

arr[0] = [0, 10] #setting the boundary conditions
arr[1] = [10, 0] #setting the boundary conditions
arrT[0] = [arr[0],      arr[1],      new_point(arr[0], arr[1]),      norm_vec(vector(arr[0],arr[1])),            0,            0,            0,           0] #setting the boundary condition so there is no need for an annoying if clause
#     starting point  end point      new point between the two       normal vector to the vector           normal vector   time_taken  end_velocity   newly created point 'new_point + norm_vec * norm_vec_factor'
#print(physics(arrT[0,6] + 1, vector(arrT[0,0], arrT[0,1])))
print(physics(-0, [10, -10]))
#print(vector(arrT[0,0], arrT[0,1]))





end_time = time.process_time()
#print('\n' * 5, f"Elapsed time: {end_time - start_time} seconds")