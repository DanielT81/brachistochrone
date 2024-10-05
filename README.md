# brachistochrone
<br>
brachistochrone replication<br><br>

work in progress<br><br>

The brachistochrone is a problem already having been precisly analytically solved revolving about 2 points in a conservative field. The goal is to optimize a curve connecting the two points based on the time a objects takes to get from one point to the other accelerated by the conservative field.<br>

Current idea is to devide the hypothetical into connected vectors and locally optimizing those. Problem with that being that there is no prove yet for the local optimization being part of the global optimization. <br>

Other plan would be to already having the direction orthogonal to the forces field lines subdevided and optimizing those with respect to speed and time. Still faces issues regarding grading of speed gained to time taken for sub-part on each vector.<br>

<br><br>
pseudo code for optimisation:<br>
<br>
given arrays: <br>
**arr**  | array with the given points <br>
1. index <br>
2. coordinates <br>


**arrT** | array with the new point currently being optimized <br>

1. index of the point the vector starts from and the index of the point it ends at
2. point the vector starts from<br>
3. point the vector goes to<br>
4. @new_point between @1 and @2<br>
5. @norm_vec to the connecting vector<br>
6. @norm_vec_factor for the normal vector<br>
7. time taken to get from @1 over @3 + @5*@4 to @2<br>
8. end velocity after the calculation
9. @4 + @5*@6

first row is for the old time that is to beat<br>
second row is for the new time that has beat<br>

<br><br><br><br>
given functions:\
**vector**    | Input: minuend point and subtrahend point | Output: numpy array with the vector difference of the points<br>
**new_point** | Input: point and vector | Output: new point half a vector from the original point<br>
**norm_vec**  | Input: vector | Output: normalized normal vector to the given vector<br>
**sort_arr**  | sorts @arr based on the entries x-coordinate<br>