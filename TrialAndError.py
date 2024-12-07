import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time

'''defining all the variables'''
changes = 0
g = -9.81
ATP = 10000 # amount of total points in the system
set_start_point = [0,10] #the set start point for the computation
set_end_point = [10,0] #the set end point for the computation
initial_x_values = np.linspace(set_start_point[0], set_end_point[0], ATP)
initial_y_values = np.linspace(set_start_point[1], set_end_point[1], ATP)
arr = np.array([initial_x_values,initial_y_values]).transpose()
last_vel = 0 # the end velocity of the previous point


'''defining all the functions'''
def sqr(def_var) -> float: # function that returns the square of a float
    return def_var ** 2


    #
def cart_norm(def_vec) -> float: # function that puts out the cartesian norm for a vector
    return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))


    #
def vec(start_point, end_point) -> np.ndarray: # function that returns the vector between two points
    result = end_point - start_point
    return result


    #
def mid_point(start_point, end_point) -> np.ndarray: # function that returns the point directly in the middle of two points
    def_point = start_point + 0.5 * vec(start_point, end_point)
    return def_point


    #
def norm_vec(def_vec) -> np.ndarray: # function that returns the normalized normal vector to a vector
    def_norm_vec = np.array([float(-def_vec[0]), float(def_vec[1])]) / float(sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))) # normalvector to def_vec
    return def_norm_vec


    #
def abc_formular(def_a: float, def_b: float, def_c: float) -> float: # returns the smallest non-negative value
    def_sol_time1 = (-def_d + sqrt(sqr(def_d) - 4 * def_c *def_e)) / (2 * def_c)
    def_sol_time2 = (-def_d - sqrt(sqr(def_d) - 4 * def_c *def_e)) / (2 * def_c)
    if def_sol_time1 and def_sol_time2 < 0:
        print(f'{def_a}, {def_b}, {def_c} both make negative times)
        break
    if def_sol_time1 < 0:
        return def_sol_time2
    if def_sol_time2 < 0:
        return def_sol_time1
    if def_sol_time1 < def_sol_time2:
        return def_sol_time1
    else:
        return def_sol_time2


    #
def physics(start_vel: float, def_start_point: np.ndarray, def_end_point: np.ndarray, *debugger) -> np.ndarray: # function that calculates the time taken for a
    # point to slide down a linear slope
    """
    :return [time_result, velocity_result] \n
    :def_start_point np.array \n
    :def_end_point np.array
    """

    if debugger:
        print(f'this is the {debugger} physics term')
    if start_vel > 0:
        print('physics: the starting velocity doesnt make sense here')
        return
    def_vec = vec(def_start_point, def_end_point)
    delta_x, delta_y = def_vec # setting the differences in coordinates
    delta_s = cart_norm(def_vec) # setting the length of the vector
    acceleration_angle_factor = sqrt(1 / (1 + sqr(delta_x / delta_y))) # factor for the acceleration based on the rolling angle
    c_coefficient = ((-0.5) * g * acceleration_angle_factor * (delta_y / sqrt(delta_y ** 2)))  # - (1/2 * 0.1 (air drag coefficient) * area of the object * velocity**2)
    d_coefficient = start_vel
    e_coefficient = delta_s
    possible_time_arr = np.roots([c_coefficient, d_coefficient, e_coefficient])
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
def calc_arr_time(def_name) -> float:
    def_time = 0
    def_vel = 0
    for def_index in range(ATP-1):
        (def_new_time, def_new_vel) = physics(def_vel, aarr(def_name[def_index]), aarr(def_name[def_index+1]))
        print(def_time, def_vel)
        def_time += def_new_time
        def_vel += def_new_vel
    print(def_time)




'''defining the variables that depend on functions'''
#global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
#optimizing_factor = np.dot(global_vec, global_vec) * 0.0001 # the global factor for the normal vector

print(physics(0,aarr([0,10]), aarr([10,0])))
calc_arr_time(arr)

#lt.plot(x, y, marker='o')  # marker='o' zeigt die Punkte an
#lt.xlabel('X-Achse')  # Beschriftung der X-Achse
#lt.ylabel('Y-Achse')  # Beschriftung der Y-Achse
#lt.title('Plot von n x 2 Array')  # Titel des Plots
#lt.grid(True)  # Gitterlinien anzeigen
#lt.show()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n \n Elapsed time: {elapsed_time} seconds")
