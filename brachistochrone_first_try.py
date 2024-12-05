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
ATI = 3  # amount of total iterations to create new points
ATP = 2 ** ATI + 1  # amount of total points in the system
arr_len = 2  # length of the non-zero values
last_vel = 0 # the end velocity of the previous point

arr = np.zeros([ATP,2])  # array with all the points and given indices to track manually
arr_time = np.zeros([3, 9], dtype=object)  # read the README to get the structure
set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
arr[0] = set_start_point
arr[1] = set_end_point


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
    global arr_len
    global arr
    print(f'arr_len has the value {arr_len}')
    sort_help = np.argsort(arr[:arr_len,0])
    arr[:arr_len] = np.array(arr[:arr_len])[sort_help]


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
    #print(f'def_vec = {def_vec} \n start_point = {start_point} \n end_point ? {end_point}')
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
    global sign
    global arr_time
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
def optimizing() -> None: # function that optimizes arr_time[1]
    global changes
    global sign
    while changes < 2:
        calc_arr_time2(optimizing_factor)
        if arr_time[2,6] < arr_time[1,6]:
            arr_time[1] = arr_time[2]
        else:
            changes += 1
            sign = sign * -1
    #print(arr_time)
    return arr_time[1,7] # return the end velocity so it can be used later on


    #
def third_layer(vel, start_point, end_point):
    global changes
    global arr_len
    changes = 0  # counts the amount of changes in the sign of the norm_vec_fac
    optimizing_factor = 0.001 * (vec(start_point, end_point) * vec(start_point, end_point)) # the difference in the norm_vec's length per step'
    calc_arr_time0(vel, start_point, end_point) # compute the original time that is to improve
    def_return_vel = optimizing() # compute the optimal arr_time[2]
    arr[arr_len] = arr_time[1,8] # adding the  new point
    arr_len += 1 # changing the length of arr because a new point got added
    #sort_arr() # sort the array afterwards
    return def_return_vel # the velocity at the end of the optimization


    #
def second_layer():
    global arr
    global last_vel
    last_vel = 0
    amount_new_points = arr_len - 1 # the amount of new points needing to be created within every second layer iteration
    for def_index, def_arr in enumerate(arr[:amount_new_points]):
        #print(f'{def_index} is def_index \n and {arr[:amount_new_points+2]} is the array \n')
        third_layer(last_vel, aarr([arr[def_index, 0],arr[def_index, 1]]), aarr([arr[def_index + 1, 0],arr[def_index + 1, 1]]))
    sort_arr()
    
    
    #
def first_layer():
    for def_index in range(ATI):
        second_layer()





'''defining the variables that depend on functions'''
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.0001 # the global factor for the normal vector



first_layer()
x = arr[:,0]
y = arr[:,1]
print('\n' *4, arr)

plt.plot(x, y, marker='o')  # marker='o' zeigt die Punkte an
plt.xlabel('X-Achse')  # Beschriftung der X-Achse
plt.ylabel('Y-Achse')  # Beschriftung der Y-Achse
plt.title('Plot von n x 2 Array')  # Titel des Plots
plt.grid(True)  # Gitterlinien anzeigen
plt.show()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n \n Elapsed time: {elapsed_time} seconds")