# brachistochrone
<br>
brachistochrone replication<br><br>

work in progress<br><br>

The brachistochrone is a problem already having been precisley analytically solved revolving about 2 points in a conservative field. The goal is to optimize a curve connecting the two points based on the time an objects takes to get from one point to the other accelerated by the conservative field.<br>

Current idea is to divide the hypothetical into connected vectors and locally optimizing those. Problem with that being that there is no prove yet for the local optimization being part of the global optimization. <br>

Other plan would be to already having the direction orthogonal to the forces field lines subdivided and optimizing those with respect to speed and time. Still faces issues regarding grading of speed gained to time taken for sub-part on each vector.<br>

<br><br>
pseudo code for optimisation:<br>
<br>
given arrays: <br>
**arr**  | array with the given points <br>
1. index <br>
2. coordinates <br>
3. time taken to get to the point starting from the previous point while having the starting velocity
4. end velocity from the first point onwards till tis point


**arrT** | array with the new point currently being optimized <br>

1. point the vector starts from<br>
2. point the vector goes to<br>
3. @new_point between @1 and @2<br>
4. @norm_vec to the connecting vector<br>
5. @norm_vec_factor for the normal vector<br>
6. time taken to get from @1 over @3 + @5*@4 to @2<br>
7. end velocity after the calculation
8. @3 + @4*@5

first row is for the old time that is to beat<br>
second row is for the new old time that is to beat<br>
third row is for the new time that has to beat<br>

<br><br><br><br>
given functions:\
**vector**    | Input: start_point and end_point | Output: numpy array with the vector difference of the points | gives the vector to get from start_point to end_point<br>
**new_point** | Input: point and vector | Output: new point half a vector from the original point<br>
**norm_vec**  | Input: vector | Output: normalized normal vector to the given vector<br>
**sort_arr**  | sorts @arr based on the entries x-coordinate<br>