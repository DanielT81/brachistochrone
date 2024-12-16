import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time

'''defining all the variables'''
changes = 0
g = np.array([0,-9.81])
ATP = 200 # amount of total points in the system
set_start_point = aarr([0,10]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
initial_x_values = np.linspace(set_start_point[0], set_end_point[0], ATP)
initial_y_values = np.linspace(set_start_point[1], set_end_point[1], ATP)
arr = np.array([initial_x_values,initial_y_values]).transpose()
new_arr = arr


'''defining all the functions'''
def sqr(def_var) -> float: # function that returns the square of a float
	return def_var ** 2
def cart_norm(def_vec) -> float: # function that puts out the cartesian norm for a vector
	return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
def vec(start_point, end_point) -> np.ndarray: # function that returns the vector between two points
	result = end_point - start_point
	return result
def vec_len(def_vec) -> float:
	def_len = sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
	return def_len
def mid_point(start_point, end_point) -> np.ndarray: # function that returns the point directly in the middle of two points
	def_point = start_point + 0.5 * vec(start_point, end_point)
	return def_point
def dot_product(def_vec1: np.ndarray, def_vec2: np.ndarray) -> np.ndarray:
	x1_sum = def_vec1[0] * def_vec2[0]
	x2_sum = def_vec1[1] * def_vec2[1]
	return x1_sum + x2_sum
def normalize_vec(def_vec) -> np.ndarray: # function that returns the normalization of a given vector
	return def_vec / cart_norm(def_vec)
def normal_vec(def_vec) -> np.ndarray: # function that returns the normalized normal vector to a vector
	def_norm_vec = normalize_vec(np.array([float(-def_vec[0]), float(def_vec[1])])) # normalvector to def_vec
	return def_norm_vec

def abc_formular_vec(def_a: np.ndarray, def_b: np.ndarray, def_c: np.ndarray, *debugger) -> list: # returns the possible times
	if debugger:
		print(debugger, ':\n')
	print(f'a: {def_a} b: {def_b} c: {def_c}')
	#print(f'a: {vec_len(def_a)} b: {vec_len(def_b)} c: {vec_len(def_c)}')
	def_sol_time1_1 = (-def_b[0] + sqrt(sqr(def_b[0]) - 4 * def_a[0] *def_c[0])) / (2 * def_a[0])
	def_sol_time1_2 = (-def_b[1] + sqrt(sqr(def_b[1]) - 4 * def_a[1] *def_c[1])) / (2 * def_a[1])
	def_sol_time2_1 = (-def_b[0] - sqrt(sqr(def_b[0]) - 4 * def_a[0] *def_c[0])) / (2 * def_a[0])
	def_sol_time2_2 = (-def_b[1] - sqrt(sqr(def_b[1]) - 4 * def_a[1] *def_c[1])) / (2 * def_a[1])

	print(f'def_sol_time1_1 = {def_sol_time1_1} \n def_sol_time1_2 = {def_sol_time1_2} \n def_sol_time2_1 = {def_sol_time2_1} \n def_sol_time2_2 = {def_sol_time2_2}')
	if abs(def_sol_time1_1 + def_sol_time1_2) > 0.001:
		#print(f'problem with def_sol_time difference')
		...
	if abs(def_sol_time2_1 + def_sol_time2_2) > 0.001:
		#print(f'problem with def_sol_time difference')
	    ...
	if abs(def_sol_time1_1 - def_sol_time2_2) > 0.001:
		print(f'def_sol_time1_1 and def_sol_time2_2 are not the same {def_sol_time1_1} and {def_sol_time2_2}')
	if abs(def_sol_time1_2 - def_sol_time2_1) > 0.001:
		print(f'def_sol_time1_2 and def_sol_time2_1 are not the same {def_sol_time1_2} and {def_sol_time2_1}')
	if def_sol_time1_1 < 0 and def_sol_time1_2 < 0:
		print(f'{def_a}, {def_b}, {def_c} both make negative times')
		return ...
	#print('\n',[def_sol_time1_1,def_sol_time1_2,def_sol_time2_1,def_sol_time2_2], '\n')
	return [def_sol_time1_1,def_sol_time1_2]

def physics(def_start_vel: np.ndarray, def_start_point: np.ndarray, def_end_point: np.ndarray, *debugger) -> np.ndarray: # function that calculates the time
	# the time value for def_start_vel = 0 and the vector [10,-10] is supposed to be 2.019s
	"""
	:return [time_result, velocity_result] \n
	:def_start_point np.ndarray \n
	:def_end_point np.ndarray
	"""

	if debugger:
		print(f'\n\n\nthis is the {debugger} physics term')
	if def_start_vel[0] < 0:
		print('physics: the starting velocity doesnt make sense here')
		return
	def_vec = vec(def_start_point, def_end_point)
	half_acceleration = 0.5 * (dot_product(g, normalize_vec(def_vec))) * normalize_vec(def_vec)
	#print('\n', half_acceleration,  normalize_vec(def_vec))


	def_start_vel = def_start_vel
	negative_def_vec = -def_vec
	def_time_results = abc_formular_vec(half_acceleration, def_start_vel, negative_def_vec)
	if abs(def_time_results[0]) != def_time_results[0]:
		def_time_result = def_time_results[1]
	elif abs(def_time_results[1]) != def_time_results[1]:
		def_time_result = def_time_results[0]
	else:
		if def_time_results[0] < def_time_results[1]:
			def_time_result = def_time_results[0]
		else:
			def_time_result = def_time_results[1]
	def_vel_result = def_start_vel + (half_acceleration * def_time_result * 2)
	return aarr([float(def_time_result), def_vel_result], dtype=object)


def calc_arr_time(def_name) -> np.ndarray:
	def_time = 0
	def_vel = aarr([0,0])
	for def_index in range(ATP-1):
		(def_new_time, def_new_vel) = physics(def_vel, aarr(def_name[def_index]), aarr(def_name[def_index+1]), def_index)
		print(def_time, def_vel)
		def_time += def_new_time #/ 2
		def_vel = def_new_vel
	#print([def_time, def_vel])
	#return np.array([def_time, def_vel], dtype=object)
	return def_time


def move_random_point() -> None: # rng point in arr gets moved by amount and stored in same column in new_arr
	global new_arr
	def_index = np.random.randint(1, ATP - 1)
	def_old_point = arr[def_index]
	def_vec = vec(arr[def_index-1],arr[def_index+1])
	def_norm_vec = normal_vec(def_vec)
	def_new_point = def_old_point + def_norm_vec * optimizing_factor * np.random.choice([1,-1])
	new_arr[def_index] = def_new_point


def loop(def_iterations: int or float) -> None:
	global arr
	global arr_time
	global new_arr
	for def_iteration in range(def_iterations):
		new_arr = arr
		move_random_point()
		new_time = calc_arr_time(new_arr)
		if new_time < arr_time:
			arr = new_arr
			arr_time = new_time
		else:
			new_arr = arr




'''defining the variables that depend on functions'''
arr_time = calc_arr_time(arr)
global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
optimizing_factor = np.dot(global_vec, global_vec) * 0.001 # the global factor for the normal vector

#print(physics(aarr([0,0]),aarr([0,10]), aarr([10,0])))
print('\n' * 5)



loop(10000)
print(arr)



'''
vecc1 = np.array([0,50])
vecc2 = np.array([10,0])
vell = np.array([10,-50])
(timee, velly) = physics(vell, vecc1, vecc2)
print(f'\n \n \nphysics term normally is {timee, velly} or {timee, vec_len(velly)}')
#print(test_val / physics(aarr([0,0]), set_start_point, set_end_point, 2)[0] - 1.9992130052793873)
'''





#lt.plot(x, y, marker='o')  # marker='o' zeigt die Punkte an
#lt.xlabel('X-Achse')  # Beschriftung der X-Achse
#lt.ylabel('Y-Achse')  # Beschriftung der Y-Achse
#lt.title('Plot von n x 2 Array')  # Titel des Plots
#lt.grid(True)  # Gitterlinien anzeigen
#lt.show()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n \n Elapsed time: {elapsed_time} seconds")
