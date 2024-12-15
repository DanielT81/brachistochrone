import time
from math import sqrt, sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray as aarr
start_time = time.perf_counter()  # to track the computation-time

'''defining all the variables'''
changes = 0
g = np.array([0,-9.81])
ATP = 1000 # amount of total points in the system
set_start_point = aarr([0,30]) #the set start point for the computation
set_end_point = aarr([10,0]) #the set end point for the computation
initial_x_values = np.linspace(set_start_point[0], set_end_point[0], ATP)
initial_y_values = np.linspace(set_start_point[1], set_end_point[1], ATP)
arr = np.array([initial_x_values,initial_y_values]).transpose()
last_vel = 0 # the end velocity of the previous point
test_val = 0

'''defining all the functions'''
def sqr(def_var) -> float: # function that returns the square of a float
	return def_var ** 2
def cart_norm(def_vec) -> float: # function that puts out the cartesian norm for a vector
	return sqrt(sqr(def_vec[0]) + sqr(def_vec[1]))
def vec(start_point, end_point) -> np.ndarray: # function that returns the vector between two points
	result = end_point - start_point
	return result
def mid_point(start_point, end_point) -> np.ndarray: # function that returns the point directly in the middle of two points
	def_point = start_point + 0.5 * vec(start_point, end_point)
	return def_point
def dot_product(def_vec1: np.ndarray, def_vec2: np.ndarray) -> np.ndarray:
	x1_sum = def_vec1[0] * def_vec2[0]
	x2_sum = def_vec1[1] * def_vec2[1]
	return x1_sum + x2_sum
def normal_vec(def_vec) -> np.ndarray: # function that returns the normalized normal vector to a vector
	def_norm_vec = normalize_vec(np.array([float(-def_vec[0]), float(def_vec[1])])) # normalvector to def_vec
	return def_norm_vec
def normalize_vec(def_vec) -> np.ndarray:
	return def_vec / cart_norm(def_vec)
def abc_formular_vec(def_a: np.ndarray, def_b: np.ndarray, def_c: np.ndarray) -> list: # returns the possible times
	print(f'{def_a} \n {def_b} \n {def_c}')
	def_sol_time1_1 = (-def_b[0] + sqrt(sqr(def_b[0]) - 4 * def_a[0] *def_c[0])) / (2 * def_a[0])
	def_sol_time1_2 = (-def_b[1] + sqrt(sqr(def_b[1]) - 4 * def_a[1] *def_c[1])) / (2 * def_a[1])
	def_sol_time2_1 = (-def_b[0] - sqrt(sqr(def_b[0]) - 4 * def_a[0] *def_c[0])) / (2 * def_a[0])
	def_sol_time2_2 = (-def_b[1] - sqrt(sqr(def_b[1]) - 4 * def_a[1] *def_c[1])) / (2 * def_a[1])

	print(f'def_sol_time1_1 = {def_sol_time1_1} \n def_sol_time1_2 = {def_sol_time1_2} \n def_sol_time2_1 = {def_sol_time2_1} \n def_sol_time2_2 = '
	      f'{def_sol_time2_2}')
	if abs(def_sol_time1_1 + def_sol_time1_2) > 0.001:
		print(f'problem with def_sol_time difference')
	if abs(def_sol_time2_1 + def_sol_time2_2) > 0.001:
		print(f'problem with def_sol_time difference')
	if def_sol_time1_1 - def_sol_time2_2 > 0.001:
		print(f'def_sol_time1_1 and def_sol_time2_2 are not the same')
	if def_sol_time1_2 - def_sol_time2_1 > 0.001:
		print(f'def_sol_time1_2 and def_sol_time2_1 are not the same')
	if def_sol_time1_1 < 0 and def_sol_time1_2 < 0:
		print(f'{def_a}, {def_b}, {def_c} both make negative times')
		return ...
	#print('\n',[def_sol_time1_1,def_sol_time1_2,def_sol_time2_1,def_sol_time2_2], '\n')
	return [def_sol_time1_1,def_sol_time1_2]

def physics(def_start_vel: np.ndarray, def_start_point: np.ndarray, def_end_point: np.ndarray, *debugger) -> np.ndarray: # function that calculates the time
	# the time value for def_start_vel = 0 and the vector [10,-10] is supposed to be 2.019s
	# the end velocity vector for def_start_vel = 0 and the vector [10,-10] is supposed to be [4.905, -4.905]
	# taken
	# for a
	# point to slide down a linear slope
	"""
	:return [time_result, velocity_result] \n
	:def_start_point np.array \n
	:def_end_point np.array
	"""

	if debugger:
		print(f'this is the {debugger} physics term')
	if def_start_vel[0] < 0:
		print('physics: the starting velocity doesnt make sense here')
		return
	def_vec = vec(def_start_point, def_end_point)
	#delta_x, delta_y = def_vec # setting the differences in coordinates
	#print('within physics is', g, (dot_product(g, normalize_vec(def_vec))),'\n')
	half_acceleration = 0.5 * (dot_product(g, normalize_vec(def_vec))) * normalize_vec(def_vec) # (normalize_vec(def_vec))) # - (1/2 * 0.1 (air drag
	# coefficient) *
	# area of
	# the
	# object *
	# velocity**2)
	#print(half_acceleration,  normalize_vec(def_vec))
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
	def_vel_result = def_start_vel + half_acceleration * def_time_result * 0.5
	return aarr([float(def_time_result), def_vel_result], dtype=object)


#
def calc_arr_time(def_name) -> float:
	global test_val
	def_time = 0
	def_vel = aarr([0,0])
	for def_index in range(ATP-1):
		(def_new_time, def_new_vel) = physics(def_vel, aarr(def_name[def_index]), aarr(def_name[def_index+1]))
		print(def_time, def_vel)
		def_time += def_new_time
		def_vel = def_new_vel
	test_val = def_time
	print(f'\n \n \n \ndef_time total = {def_time}')
	print(f'def_vel total = {def_vel}')
	return [def_time, def_vel]


	#
def move_random_point(def_name1: np.ndarray, def_name2: np.ndarray) -> None:
	def_index = np.random.randint(1, ATP - 1)
	def_old_point = def_name1[def_index]
	def_vec = vec(def_name1[def_index-1], def_name1[def_index+1])
	def_norm_vec = normal_vec(def_vec)
	def_new_point = def_old_point + def_norm_vec * 0.001 * np.random.choice([1,-1])
	def_name2[def_index] = def_new_point


	#






'''defining the variables that depend on functions'''
#global_vec = vec(set_start_point, set_end_point) # setting the boundary vector for easy debugging
#optimizing_factor = np.dot(global_vec, global_vec) * 0.0001 # the global factor for the normal vector

#print(physics(aarr([0,0]),aarr([0,10]), aarr([10,0])))
calc_arr_time(arr)
print(f'\n \n \nphysics term normally is {physics(aarr([0,0]), set_start_point, set_end_point)}')
print(test_val / physics(aarr([0,0]), set_start_point, set_end_point)[0] - 1.9992130052793873)
#lt.plot(x, y, marker='o')  # marker='o' zeigt die Punkte an
#lt.xlabel('X-Achse')  # Beschriftung der X-Achse
#lt.ylabel('Y-Achse')  # Beschriftung der Y-Achse
#lt.title('Plot von n x 2 Array')  # Titel des Plots
#lt.grid(True)  # Gitterlinien anzeigen
#lt.show()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\n \n Elapsed time: {elapsed_time} seconds")
