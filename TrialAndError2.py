import numpy as np
from math import sin, cos, pi, sqrt, atan
from numpy import asarray as aarr

g = -9.81
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
def physics(start_vel, start_point, end_point, *debugger):
    if debugger:
        print(f'this is the {debugger} physics term')
    if start_vel > 0:
        print('physics: the starting velocity doesnt make sense here')
        return
    delta_x = vec(start_point, end_point)[0]
    delta_y = vec(start_point, end_point)[1]
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


length = 10000

arr = np.empty(shape=(3,8), dtype=object)
good_naughty = np.zeros(shape=(2,length), dtype=object)



def optimizing(vel, start_point, end_point):
    boundary_vec = vec(start_point, end_point)
    newpoint = new_point(start_point, end_point)
    def_vec = vec(start_point, end_point)
    normvec = norm_vec(vec(start_point, end_point))
    norm_vec_fac = 0
    optimizing_factor = 1
    arr[0] = [start_point, end_point, newpoint, normvec, norm_vec_fac, physics(vel, start_point, end_point)[0], physics(vel, start_point, end_point)[1], 0]

    for i in range(length):
        if ((physics(vel, start_point, new_point(start_point, end_point) + normvec * (i-5000) * 0.001) + physics(physics(vel, start_point, new_point(start_point, end_point) + normvec)[1], new_point(start_point, end_point) + normvec * (i - 5000) * 0.001, end_point))[0]) < physics(vel, start_point, end_point)[0]:
            good_naughty[0, i] = new_point(start_point, end_point) + normvec * (i-5000) * 0.001
        else:
            good_naughty[1, i] = new_point(start_point, end_point) + normvec * (i-5000) * 0.001

    '''
    while abs(optimizing_factor) > 0.01:
        if physics(vel, start_point, )
    '''

p1 = [10, 0]
p2 = [0, 10]

i = 6000

#print(physics(0, p1, new_point(p1, p2) + norm_vec(vec(p1,p2)) * (i-5000) * 0.001)[0])
#print(   physics(0, p1, new_point(p1, p2) + norm_vec(vec(p1,p2)) * (i-5000) * 0.001)[1]) #end velocity after reaching the first sub-point
#print( new_point(p1, p2) + norm_vec(vec(p1,p2)) * (i-5000) * 0.001, p2   )

'''
with open('output.txt', 'w') as file:
    # Print to the file instead of the console
    for i in range(len(good_naughty)):
        print(good_naughty[:, i], file=file)
        

optimizing(0, p1, p2)
for i in range(10000):
    print(good_naughty[:, i])
'''
# Open the file in write mode
print(physics(0, p1, new_point(p1, p2) + norm_vec(vec(p1,p2)) * (i-5000) * 0.001)[1])
print(physics(0, p1, p2))




#print(vec(p1, p2)[0])
