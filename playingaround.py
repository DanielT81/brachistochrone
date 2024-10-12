import time
start_time = time.process_time()  # to track the computation-time
for i in range(1000000):
    pass
import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as aarr
import scipy as sp




g = -9.81
index_number = 2
AIP = 4  # amount of iterations to create new points
ATP = 2 ** AIP + 1  # amount of total points in the system
arr_len = 2  # length of the non-zero values


def sqr(var):
    return var ** 2


    #
def vec(start_point, end_point):
    result = aarr(start_point) - aarr(end_point)
    return result


    #
def new_point(start_point, end_point):
    def_point = aarr(start_point) + 0.5 * aarr(vec(end_point, start_point))
    return def_point


    #
def sort_arr():
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
def norm_vec(def_vec):
    def_norm_vec = np.array([float(-def_vec[0]), float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1])))
    return def_norm_vec


    #
def optimizer(start_vel, start_point, end_point):
    boundary_vec = vec(start_point, end_point)
    new_point = new_point(start_point, end_point)
    def_vec = vec(start_point, end_point)
    norm_vec = norm_vec(def_vec)
    norm_vec_fac = 0
    start_vel = arr_time[0, 7]
    arr_time[1] = [start_point, end_point, new_point, norm_vec, norm_vec_fac, physics(vel, start_point)]


    #
def physics(start_vel, def_vec, *debugger):
    if debugger:
        print(f'this is the {debugger} physics term')
    if start_vel > 0:
        print('physics: the starting velocity doesnt make sense here')
        return
    delta_x = def_vec[0]
    delta_y = def_vec[1]
    delta_s = sqrt(sqr(delta_x) + sqr(delta_y))
    acceleration_angle_factor = sqrt(1 / (1 + sqr(delta_x / delta_y)))
    a_coefficient = ((-0.5) * g * acceleration_angle_factor * (delta_y / sqrt(delta_y ** 2)))  # - (1/2 * 0.1 (air drag coefficient) * area of the object * velocity**2)
    b_coefficient = start_vel
    c_coefficient = delta_s
    possible_time_arr = np.roots([a_coefficient, b_coefficient, c_coefficient])
    # if str(t[0])[len(t[0])] == 'j' or str(t[1])[len(t[1])] == 'j':
    if type(possible_time_arr[0]) == np.complex128 or type(possible_time_arr[1]) == np.complex128:
        print('physics: problem with the imaginary unity')
        return
    possible_time_arr = possible_time_arr[np.argsort(possible_time_arr)]
    if possible_time_arr[1] < 0:
        print('physics: both t values are negative')
        print(possible_time_arr)
        return
    if possible_time_arr[0] > 0:
        print('physics: both t values are positive')
        print(possible_time_arr)
        return
    time_result = possible_time_arr[1]
    velocity_result = start_vel + a_coefficient * time_result
    #print([time_result, velocity_result])
    return [time_result, velocity_result]


    #
def vec_arr():
    iterations = 0


arr = np.zeros([ATP, 4])  # array with all the points and given indices to track manually
arr_time = np.zeros([3, 8], dtype=object)  # read the README to get the structure

arr[0] = [0, 10]  # setting the boundary conditions
arr[1] = [10, 0]  # setting the boundary conditions
boundary_vec = vec(arr[0], arr[1])
arr_time[0] = [arr[0],
               arr[1],
               new_point(arr[0], arr[1]),
               norm_vec(boundary_vec),
               0,
               physics(0, boundary_vec)[0],
               physics(0, boundary_vec)[1],
               new_point(arr[0], arr[1])]  # setting the boundary condition so there is no need for an annoying if-clause
# print(physics(arrT[0,6] + 1, vector(arrT[0,0], arrT[0,1])))



print(vec([0, 10], [10, 0]), '\n')
physics(0, [10, -10])
#physics(0,vector(arr[1], arr[0]))


end_time = time.process_time()
print('\n' * 3, f"Elapsed time: {end_time - start_time} seconds")