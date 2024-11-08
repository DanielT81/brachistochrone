import time
start_time = time.perf_counter()  # to track the computation-time
import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as aarr
import scipy as sp



set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation


g = -9.81 # gravitational acceleration factor
index_number = 2 # setting the current number of indices 
AIP = 4  # amount of iterations to create new points
ATP = 2 ** AIP + 1  # amount of total points in the system
arr_len = 2  # length of the non-zero values
optimizing_factor = 0.001 # the difference in the norm_vec's length per step'



def sqr(var) -> float: # function that returns the square of a float
    return var ** 2


    #
def vec(start_point, end_point) -> np.array: # function that returns the vector between two points
    result = end_point - start_point
    return result


    #
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging




def mid_point(start_point, end_point) -> np.array: # function that returns the point directly in the middle of two points
    def_point = start_point + 0.5 * vec(start_point, end_point)
    return def_point


    #
def norm_vec(def_vec) -> np.array: # function that returns the normalized normal vector to a vector
    def_norm_vec = np.array([float(-def_vec[0]), float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1])))
    return def_norm_vec


    #
def sort_arr() -> None: # function that just sorts the array arr depending on the x-coordinate
    iterations = 0
    total_iterations = np.size(arr, 0)
    while iterations <= total_iterations:
        if arr[iterations, 1] != 0.0:
            iterations += 1
        else:
            arr[:iterations] = arr[np.argsort(arr[:iterations, [1][0]])]
            arr_len = iterations
            break


    #
def physics(start_vel, def_vec, *debugger) -> list: # function that calculates the time taken for a point to roll down a vector
    if debugger:
        print(f'this is the {debugger} physics term')
    if start_vel > 0:
        print('physics: the starting velocity doesnt make sense here')
        return
    delta_x, delta_y = def_vec # setting the differences in coordinates
    delta_s = sqrt(sqr(delta_x) + sqr(delta_y)) # setting the length of the vector
    acceleration_angle_factor = sqrt(1 / (1 + sqr(delta_x / delta_y))) # factor for the acceleration based on the rolling angle
    a_coefficient = ((-0.5) * g * acceleration_angle_factor * (delta_y / sqrt(delta_y ** 2)))  # - (1/2 * 0.1 (air drag coefficient) * area of the object * velocity**2)
    b_coefficient = start_vel
    c_coefficient = delta_s
    possible_time_arr = np.roots([a_coefficient, b_coefficient, c_coefficient])
    # if str(t[0])[len(t[0])] == 'j' or str(t[1])[len(t[1])] == 'j':
    if type(possible_time_arr[0]) == np.complex128 or type(possible_time_arr[1]) == np.complex128:
        print('physics: problem with the imaginary unity',  debugger)
        return
    possible_time_arr = possible_time_arr[np.argsort(possible_time_arr)]
    if possible_time_arr[1] < 0:
        print('physics: both t values are negative',  debugger)
        print(possible_time_arr)
        return
    if possible_time_arr[0] > 0:
        print('physics: both t values are positive',  debugger)
        print(possible_time_arr)
        return
    time_result = possible_time_arr[1]
    velocity_result = start_vel + a_coefficient * time_result
    #print([time_result, velocity_result])
    return [time_result, velocity_result]
    
    
    
    #
def calc_arr_time_row() -> None:
    start_vel = arr_time[1,0]
    (start_point, end_point) = arr_time[1, 1:3]
    mid_point = arr_time[1,3]
    norm_vec = arr_time[1,4]
    norm_vec_fac = arr_time[1,5] + optimizing_factor
    new_point = mid_point + norm_vec * norm_vec_fac
    (def_time1, def_vel1) = physics(start_vel, vec(start_point, new_point))
    (def_time2, def_vel2) = physics(def_vel1, vec(new_point, end_point))
    
    arr_time[2] = [start_vel, start_point, end_point, 
    mid_point, norm_vec, norm_vec_fac, 
    def_time1+def_time2, def_vel2, new_point]
    
    
    #
'''defining the arrays that will help calculate'''
    
arr = np.zeros([ATP, 3], dtype=object)  # array with all the points and given indices to track manually
arr_time = np.zeros([3, 9], dtype=object)  # read the README to get the structure

arr[0] = [set_start_point,0,0]  # setting the boundary conditions
arr[1] = [set_end_point, *physics(0,vec(set_start_point,set_end_point))]# setting the boundary conditions
boundary_vec = vec(set_start_point, set_end_point)
arr_time[0] = [
               0,
               set_start_point,
               set_end_point,
               mid_point(set_start_point, set_end_point),
               norm_vec(boundary_vec),
               0,
               *physics(0, boundary_vec),
               mid_point(set_start_point, set_end_point)]  # setting the boundary condition so there is no need for an annoying if-clause

    
    
    
    
    
    
    
good_naughty = np.zeros([])

def optimizing(vel, start_point, end_point):
    number_of_optimization_steps = int(sqrt(np.dot(start_point + end_point, start_point + end_point) / optimizing_factor)) # minimum amount of steps
    start_vel = vel
    midpoint = mid_point(start_point, end_point)
    def_vec = vec(start_point, end_point)
    normvec = norm_vec(def_vec)
    norm_vec_fac = 0
    arr_time[0] = [start_vel, start_point, end_point, midpoint, normvec, norm_vec_fac, *physics(vel, def_vec), 0]
    
    
    def def_vector(index):
        def_vector = ...
    
    
    '''
    for i in range(10000):
        if ((physics(vel,  vec(mid_point(start_point, end_point) + normvec * (i-5000) * 0.001) + physics(physics(vel, global_vec(global_vec) + normvec)[1],  mid_point(start_point, end_point) + normvec * (i - 5000) * 0.001, end_point))[0])) < physics(vel, global_vec)[0]:
            good_naughty[0, i] = mid_point(start_point, end_point) + normvec * (i-5000) * 0.001
        else:
            good_naughty[1, i] = mid_point(start_point, end_point) + normvec * (i-5000) * 0.001
    '''

    '''
    while abs(optimizing_factor) > 0.01:
        if physics(vel, start_point, )
    '''


optimizing(0, set_start_point, set_end_point)
print(arr_time, '\n'*2, arr)


#print(good_naughty)