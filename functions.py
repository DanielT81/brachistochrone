print('\n' * 2)
import time
start_time = time.perf_counter()  # to track the computation-time
import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as aarr
from matplotlib import pyplot as plt


index_number = 2 # setting the current number of indices
AIP = 4  # amount of iterations to create new points
ATP = 2 ** AIP + 1  # amount of total points in the system
arr_len = 2  # length of the non-zero values
arr = np.zeros([ATP, 3], dtype=object)  # array with all the points and given indices to track manually
arr_time = np.zeros([3, 9], dtype=object)  # read the README to get the structure

g = -9.81

optimizing_factor = 0.001
changes = 0
sign = -1


def sqr(var) -> float: # function that returns the square of a float
    return var ** 2


    #
def vec(start_point, end_point) -> np.array: # function that returns the vector between two points
    result = end_point - start_point
    return result


    #
def mid_point(start_point, end_point) -> np.array: # function that returns the point directly in the middle of two points
    def_point = start_point + 0.5 * vec(start_point, end_point)
    return def_point


    #
def norm_vec(def_vec) -> np.array: # function that returns the normalized normal vector to a vector
    def_norm_vec = np.array([float(-def_vec[0]), float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))) # normalvector to def_vec
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
def physics(start_vel, def_vec, *debugger) -> np.array: # function that calculates the time taken for a point to roll down a vector
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
    return aarr([time_result, velocity_result])


    #
def calc_arr_time0(def_vel, start_point, end_point) -> None: # function that calculates the first and second row of
    def_vel = def_vel
    start_point, end_point = start_point, end_point
    def_mid_point = mid_point(start_point, end_point)
    def_vec = vec(start_point, end_point)
    def_norm_vec = norm_vec(def_vec)
    new_point = def_mid_point
    def_time1, def_vel1 = physics(def_vel, vec(start_point, new_point))
    def_time2, def_vel2 = physics(def_vel1, vec(new_point, end_point))
    arr_time[0] = [def_vel, # start velocity
                   start_point, # start point
                   end_point, # end point
                   def_mid_point, # middle point
                   def_norm_vec, # normal vector
                   0, # normal vector factor
                   def_time1+def_time2, # time taken
                   def_vel1, # end velocity
                   new_point] # newly created point
    arr_time[1] = arr_time[0]


    #
def calc_arr_time2(def_index) -> None: # function that calculates the third row of arr_time based on the second row
    start_vel = arr_time[1,0]
    (start_point, end_point) = arr_time[1, 1:3]
    def_mid_point = arr_time[1,3]
    def_norm_vec = arr_time[1,4]
    def_norm_vec_fac = arr_time[1,5] + optimizing_factor
    new_point = def_mid_point + def_norm_vec * def_norm_vec_fac * sign
    def_time1, def_vel1 = physics(start_vel, vec(start_point, new_point))
    def_time2, def_vel2 = physics(def_vel1, vec(new_point, end_point))

    arr_time[2] = [start_vel,
                   start_point,
                   end_point,
                   def_mid_point,
                   def_norm_vec,
                   def_norm_vec_fac,
                   def_time1+def_time2,
                   def_vel2,
                   new_point]


    #
def comp_time() -> None:
    if arr_time[2, 6] < arr_time[1, 6]:
        arr_time[1] = arr_time[2]
    else:
        global changes
        changes += 1
        print(changes)
        global sign
        sign = sign * -1
