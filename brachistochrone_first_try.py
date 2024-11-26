from functions import *

changes = 0
sign = -1
set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.0001


def comp_time2() -> None:
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


def optimizing(vel, start_point, end_point):
    global changes
    changes = 0  # counts the amount of changes in the sign of the norm_vec_fac
    optimizing_factor = 0.001 * (vec(start_point, end_point) * vec(start_point, end_point)) # the difference in the norm_vec's length per step'
    calc_arr_time0(vel, start_point, end_point)

    comp_time2()
    #while changes <= 2:
    #    calc_arr_time2(optimizing_factor)
    #    comp_time()


print('first changes: ' + str(changes))
optimizing(0, set_start_point, set_end_point)
print(arr_time[0,6], ' original time \n', arr_time[2,6], ' new time \n')
