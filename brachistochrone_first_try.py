from functions import *

set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.0001


def optimizing(vel, start_point, end_point):
    global changes
    changes = 0  # counts the amount of changes in the sign of the norm_vec_fac
    optimizing_factor = 0.001 * (vec(start_point, end_point) * vec(start_point, end_point)) # the difference in the norm_vec's length per step'
    calc_arr_time0(vel, start_point, end_point)
    while changes <= 2:
        calc_arr_time2(optimizing_factor)
        comp_time()



optimizing(0, set_start_point, set_end_point)
print(arr_time[0,6], ' original time \n', arr_time[2,6], ' new time \n')
