import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time

'''defining all the variables'''
changes = 0
sign = -1
g = -9.81
index_number = 2 # setting the current number of indices
AIP = 4  # amount of iterations to create new points
ATP = 2 ** AIP + 1  # amount of total points in the system
arr_len = 2  # length of the non-zero values

arr = np.zeros([ATP, 3], dtype=object)  # array with all the points and given indices to track manually
arr_time = np.zeros([3, 9], dtype=object)  # read the README to get the structure
set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation


'''defining all the functions'''
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
def comp_time() -> None:
    global changes
    global sign
    while changes < 2:
        print('\n'*5, arr_time[1:3], '\n'*5)
        calc_arr_time2(optimizing_factor)
        if arr_time[2,6] < arr_time[1,6]:
            arr_time[1] = arr_time[2]
        else:
            changes += 1
            print(f'changes: {changes}')
            sign = sign * -1
            print(f'sign in comp_time2 {sign}')
    print(arr_time)


    #
def calc_arr_time2(def_index) -> None: # function that calculates the third row of arr_time based on the second row
    global sign
    global arr_time
    start_vel = arr_time[1,0]
    (start_point, end_point) = arr_time[1, 1:3]
    def_mid_point = arr_time[1,3]
    def_norm_vec = arr_time[1,4]
    def_norm_vec_fac = arr_time[1,5] + optimizing_factor
    new_point = def_mid_point + def_norm_vec * def_norm_vec_fac * sign
    print(f'sign: {sign}')
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
def optimizing(vel, start_point, end_point):
    global changes
    changes = 0  # counts the amount of changes in the sign of the norm_vec_fac
    optimizing_factor = 0.001 * (vec(start_point, end_point) * vec(start_point, end_point)) # the difference in the norm_vec's length per step'
    calc_arr_time0(vel, start_point, end_point)
    comp_time()


'''defining the variables that depend on functions'''
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.0001



print('first changes: ' + str(changes))
optimizing(0, set_start_point, set_end_point)
print(arr_time[0,6], ' original time \n', arr_time[2,6], ' new time \n')



end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")