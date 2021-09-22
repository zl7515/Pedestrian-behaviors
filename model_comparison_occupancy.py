#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
import time
#To parallelize the code
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
pi = np.pi
np.random.seed(123)
#------------------------------------------------------------------------------#
#To precompile certain functions in order to make them faster - unsure about speedup at present
# from numba import jit, double, int8
#------------------------------------------------------------------------------#
def distance_to_wall(i,w):
    """ Computes distance from person i to wall w """

    #For clarity: person, i, values
    xi = x[0][i]
    yi = x[1][i]
    ri = r[i]

    #For clarity: wall, w, values
    wall = walls[:,w]
    a = wall[0]
    b = wall[1]
    c = wall[2]
    wall_start = wall[3]
    wall_end = wall[4]

    #Extract start and end points, (x,y), of wall
    if b != 0:
        x1 = wall_start
        x2 = wall_end
        y1 = - (a * x1 + c) / b
        y2 = - (a * x2 + c) / b
    else:
        x1 = x2 = - c / a
        y1 = wall_start
        y2 = wall_end

    #Compute distance from person i to wall
    tx = x2-x1
    ty = y2-y1
    val =  ((xi - x1) * tx + (yi - y1) * ty) / (tx*tx + ty*ty)
    if val > 1:
        val = 1
    elif val < 0:
        val = 0
    x_val = x1 + val * tx
    y_val = y1 + val * ty
    dx = x_val - xi
    dy = y_val - yi
    dist = math.sqrt(dx*dx + dy*dy)

    return dist
#------------------------------------------------------------------------------#
#So far no speed gained by jit-ing this function
# @jit(double(double,double,double))
def smallest_positive_quadroot(a,b,c):
    """ Significantly aster than numpy.roots for this case
    Returns the smallest positive root of the quadratic equation a*x^2 + b*x + c = 0.
    If there is no positive root then -1 is returned """

    d = (b**2) - (4.*a*c)
    if d < 0:
        return -1
    if d == 0:
        root = -b / (2*a)
        if root > 0:
            return root
        else:
            return -1
    else:
        sqrt_d = math.sqrt(d)
        root1 = (-b + sqrt_d) / (2*a)
        root2 = (-b - sqrt_d) / (2*a)
        if root1 > 0:
            if root2 > 0:
                if root1 < root2:
                    return root1
                else:
                    return root2
            return root1
        elif root2 > 0:
            return root2
        else:
            return -1
#------------------------------------------------------------------------------#
def wrap_angle(angle):
    """ Wraps values of angles to be in [-pi,pi]
    Can take an array as an arguement as well """
    return ( ( angle + pi) % (2. * pi ) - pi )
#------------------------------------------------------------------------------#
#Not sure if this speedup in significant: @jit
def f_w( alpha, i, wall, xi, yi, v_0i, ri, d_max, v_xi, v_yi):
    """ Compute the distance to collision with walls"""

    #Extract values from the wall array for clarity
    a = wall[0]
    b = wall[1]
    c = wall[2]
    wall_start = wall[3] - ri
    wall_end = wall[4] + ri

    #Deal with case when velocity is zero
    d = a*v_xi + b*v_yi  #numerator of delta_t
    if d == 0:
        return d_max

    #Check if there is interception with the wall in direction alpha and if not return d_max
    #For horizontal walls
    if a == 0:
        if v_yi == 0:
            return d_max
        else:
            m = v_xi/v_yi
            if m>0:
                delta_y = (c / b) - yi - ri
            else:
                delta_y = (c / b) - yi + ri
            x_intercept = xi + delta_y * m
            if x_intercept < wall_start or x_intercept > wall_end:
                return d_max
    #For vertical walls
    if b == 0:
        if v_xi == 0:
            return d_max
        else:
            m = v_yi/v_xi
            if m>0:
                delta_x = (c / a) - xi - ri
            else:
                delta_x = (c / a) - xi + ri
            y_intercept = yi + delta_x * m
            if y_intercept < wall_start or y_intercept > wall_end:
                return d_max
    #1. Need to have one for diagonal walls as well.
    #2. Can perhaps speedup process by utilizing above calculated values for returning f_alpha

    if d < 0:
        #Unsure whether to just return d_max or convert wall intergers
        return d_max
        """ Need to check if I just return d_max or should I swap the negatives and positives
        a = -a
        b = -b
        c = -c
        d = -d
        """

    delta_tvals = [None] * 2
    #Solve for time to collision with wall
    
    r_component = ri*math.sqrt(a**2 + b**2)
    abc_component = - a*xi - b*yi - c
    delta_tvals[0] = (   r_component + abc_component) / d
    delta_tvals[1] = ( - r_component + abc_component) / d

    #Pick the smallest positive delta_t value and return d_max for no positive values
    if max(delta_tvals) < 0:
        return d_max
    else:
        delta_tvals=np.array(delta_tvals)
        delta_t = np.min(delta_tvals[delta_tvals > 0])

    #Compute and return distance to collision
    dist = v_0i * delta_t
    f_alpha = min(dist,d_max)
    return f_alpha
#------------------------------------------------------------------------------#
def f_j(alpha,i,j,xi,xj,yi,yj,d_max,ri,rj,v_0i,v_xj,v_yj,v_xi, v_yi):
    """ Compute the distance to colision with person j"""

    #Deals with cases when quadratic does not need to solved
    #Contained in a separate function (below) to try improve speeds using jit but so far to no avail
    bypass = bypass_funcs(i,j,xi,xj,yi,yj,ri,rj,alpha,d_max,v_0i,v_xj,v_yj)

    if bypass != -1:
        return bypass

    #Minimize floating point operations by using temp variables
    v_xdiff = v_xj - v_xi
    v_ydiff = v_yj - v_yi
    xdiff = xj - xi
    ydiff = yj - yi

    #Coefficients of the quadratic equation to be solved
    quad_A = (v_xdiff)**2 + (v_ydiff)**2
    quad_B = 2*(v_xdiff)*(xdiff) + 2*(v_ydiff)*(ydiff)
    quad_C = (xdiff)**2 + (ydiff)**2 - (ri+rj)**2
    #Solve the Quadratic for the smallest positive root
    delta_t = smallest_positive_quadroot(quad_A, quad_B, quad_C)

    #Find and return f_alpha if positive root exists and d_max if not
    if delta_t > 0:
        dist = v_0i*delta_t
        f_alpha = min(dist,d_max)
        return f_alpha
    else:
        return d_max
#------------------------------------------------------------------------------#
#No speedup: @jit
def bypass_funcs(i,j,xi,xj,yi,yj,ri,rj,alpha,d_max,v_0i,v_xj,v_yj):

    #Check whether in contact and if so then return d_max or 0 depending on alpha
    dist = gap[i][j]  #gap is distance matrix

    if (dist <= ri + rj) and dist != 0:
        if dist < ri + rj:
            return 0

        #Check angle of possible movement - currently set to 90'
        b_delta = math.asin((rj+ri)/dist)
        b_direction = np.arctan2(yj - yi, xj - xi)

        #Lower and upper bounds for obstructed horizon
        b1, b2 = wrap_angle( np.array([ b_direction - b_delta, b_direction + b_delta]) )

        #If alpha points towards j then return 0 else return d_max
        if (alpha > b1 or alpha < b2):
            return 0
        else:
            return d_max

    #Deal with case when person i doesn't want to move and j is stationary
    if v_0i == 0 and v_xj == 0 and v_yj == 0:
        return 0

    #If none of the special cases apply return a negative value
    return -1
#------------------------------------------------------------------------------#
def f(alpha,i):
    """ Compute the minimum distance to collision in this direction"""

    #Compute movement in direction alpha at a comfortable walking speed
    v_xi = math.cos(alpha)*v_0[i]
    v_yi = math.sin(alpha)*v_0[i]

    #Find distance to collisions with persons
    f_persons = d_max
    for j in range(n):
        if ( i!=j ):
            f_persons = min( f_persons, f_j( alpha, i, j, x[0][i], x[0][j], x[1][i], x[1][j], d_max, r[i], r[j], v_0[i], v[0][j], v[1][j], v_xi, v_yi) )

    #Find distance to collisions with walls
    f_walls = d_max
    for w in range(n_walls):
        f_walls = min( f_walls , f_w( alpha, i, walls[:,w], x[0][i], x[1][i], v_0[i], r[i], d_max, v_xi, v_yi) )

    #Choose the smallest distance to collision
    f_alpha = min( f_persons , f_walls)

    #If collision is further than target destination in direction of alpha then set
    #f_alpha to d_max so that distance function gives a value of 0 in this direction
    if abs(alpha - alpha_0[i]) <= ar/2:
        d_des = math.hypot(x[0][i] - o[0][i], x[1][i] - o[1][i])
        if d_des < f_alpha:
            f_alpha = d_max

    return f_alpha
#------------------------------------------------------------------------------#
#To profile speed of the code line by line: @profile
def compute_alpha_des(i, display_falpha = False, display_distancefunc = False):
    """ Compute the minimum distance function to find alpha_des over
    the horizon of alpha values"""

    #distance to persons
    for j in range(i,n):
        gap[j][i] = gap[i][j] = np.linalg.norm ([ x[0][i] - x[0][j], x[1][i] - x[1][j] ])
        if gap[i][j] < r[i] + r[j]:
            contact_p[j][i] = contact_p[i][j] = gap[i][j]
        else:
            contact_p[j][i] = contact_p[i][j] = 0

    #distance to walls
    for w in range(n_walls):
        contact_w[i][w] = distance_to_wall(i,w)
        if contact_w[i][w] >= r[i]: 
            contact_w[i][w] = 0
    #Set the range of alphas to compute f_alpha over
    alphas = np.arange(alpha_current[i] - H[i], alpha_current[i] + H[i], ar)
    #Make sure alpha values are between 0 and 2*pi
    alphas = wrap_angle(alphas)
    #Compute the values for each of the alphas
    if (n>3):
        if __name__ == '__main__':
            #In parallel for enough pedestrians
            f_alphas = Parallel(n_jobs=num_cores)(delayed(f)(alpha,i) for alpha in alphas)
    else:
        #In serial for few pedestrians
        f_alphas = np.zeros(len(alphas))
        for index in range(len(alphas)):
            f_alphas[index] = f(alphas[index],i)    
    
    

    #Distance function for given alphas and f_alphas
    distances = d_max ** 2 + np.power(f_alphas,2) - 2. * d_max * np.multiply(f_alphas, np.cos(alpha_0[i] - alphas))

    #Set alpha_des to minimum of value given by the distance function
    min_distance_index = np.argmin(distances)
    alpha_out = alphas[min_distance_index]
    f_alpha_out = f_alphas[min_distance_index]

    #If plots are asked for
    if display_falpha:
        plot_f_alpha(i, alphas, f_alphas)
    if display_distancefunc:
        plot_distance_func(i, alphas, distances)

    #Format and return output
    result = [alpha_out, f_alpha_out]

    return result
#------------------------------------------------------------------------------#
def plot_f_alpha(i, alphas, f_alphas):
    """Plot the function f(alpha) over values of alpha for i"""
    plt.figure()
    plt.plot(alphas,f_alphas)
    plt.title("f(alpha) for i = %d" %(i))

def plot_distance_func(i, alphas, distances):
    """Plot the distance over values of alpha for i"""
    plt.figure()
    plt.title("Distance for i = %d" %(i))
    plt.plot(alphas,distances)
    plt.figure()
#------------------------------------------------------------------------------#
def compute_iacceleration(vi_x, vi_y, vdes, alphades, tau):
    """ Returns acceleration caused by direction of travel wanted by person i """
    vd_x = math.cos(alphades) * vdes
    vd_y = math.sin(alphades) * vdes
    ax = (vd_x - vi_x) / tau
    ay = (vd_y - vi_y) / tau
    return [ax,ay]
#------------------------------------------------------------------------------#
def compute_bodycollision_acceleration(i):
    """ Returns acceleration in [x,y] directions caused by body collisions
        for person i current positions of persons """

    axt = 0
    ayt = 0
    ri = r[i]
    #Collisions due to persons
    for j in range(n):
        if contact_p[i][j] != 0:
            kg = k * (ri + r[j] - contact_p[i][j])
            nx = x[0][i] - x[0][j]
            ny = x[1][i] - x[1][j]
            size_n = math.hypot(nx,ny)
            nx = nx / size_n
            ny = ny / size_n
            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay
    #Collisions due to walls
    for w in range(n_walls):
        if contact_w[i][w] != 0:
            kg = k * (ri - contact_w[i][w])
            #find normal direction to wall
            wall = walls[:,w]
            a = wall[0]
            b = wall[1]
            c = wall[2]
            wall_start = wall[3] - ri
            wall_end = wall[4] + ri
            if a == 0:
                nx = 0
                if x[1][i] >= -c/b:
                    ny = 1
                else:
                    ny = -1
            elif b == 0:
                if x[0][i] >= -c/a:
                    nx = 1
                else:
                    nx = -1
                ny = 0
            else:
                '''Add direction of normal vector based on location of person i'''
                nx = 1
                ny = b / a
                ntot = math.sqrt( nx * nx + ny * ny )
                nx = nx / ntot
                ny = ny / ntot
            '''Deal with case when end of the wall point is in contact with a person
             as it changes the normal direction of the contact force with the wall'''
            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay

    return [axt, ayt]
#------------------------------------------------------------------------------#
def plot_current_positions(fig_name, colors = None):
    """ Plots current positions of persons """
    #If colors are not inputted for the persons then blue is chosen by default
    if colors is None: colors = ['blue'] * n
    for i in range(n):
        circle = plt.Circle( (x[0][i], x[1][i]), r[1], color = colors[i])
        fig_name.gca().add_artist(circle)
    plt.title('Time = %.3f' %(t))

def plot_trajectories():
    """ Plots trajectories taken of persons """
    for i in range(n):
        plt.plot(x_full[0,i,:],x_full[1,i,:],'k')
#------------------------------------------------------------------------------#
def initialize_global_parameters():
    """ To intialize global parameters that are dependent on initial conditions or default settings preffered """
    global variablesready
    if variablesready:
        #Makes sure the scopes of the variables are global to the module
        global alpha_0, x_full, gap, H, alpha_current, alpha_des, f_alpha_des, v_des, contact_p, contact_w, n_walls
        global r, mass, v, v_0, v_full

        #angle to destination
        alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
        #Initalize the array that stores movement values over time
        x_full = np.copy(x)
        gap = np.zeros((n,n))
        #Field of Vision for each of the pedestrians
        H = np.random.uniform(H_min,H_max,n)
        #set initial alpha_direction to alpha_0
        alpha_current = np.copy(alpha_0)
        alpha_des = np.zeros(n)
        f_alpha_des = np.zeros(n)
        #Array to store v_des
        v_des = np.zeros(n)
        #Store information about persons in contact with people and walls

        if n_walls is None:
            n_walls = 0
        contact_p = np.zeros((n,n))
        contact_w = np.zeros((n,n_walls))

        if np.shape(mass) != (n,):
            mass = np.random.uniform(60,100,n)
        #Radius, r = mass/320
        r = mass/320

        #If starting starting velocities are not specified then its assumed that they are zero for all people
        if np.shape(v) != (2,n):
            v = np.zeros((2,n))
        v_full = np.copy(v)

        if np.shape(v_0) != (n,):
            v_0 = 1.3*np.ones(n)

        #For clarity
        variablesinitialized = True
        if instructions: print ("%d cores in use" %(num_cores))
    else:
        if instructions: print ("Not all required variables initialized and checked. To not avoid checking manually configure variablesready to True")
#------------------------------------------------------------------------------#
def check_model_ready():
    """ Make sure all neccessary parameters for the model are initalized properly and allows user to call initialize_global_parameters() """

    global variablesready

    if n is None:
        if instructions: print ("value of n not given")
    else:
        variablesready = True

        if x is None or np.shape(x) != (2,n):
            if instructions: print ("position values array, x, not initalized or not in the right shape (2xn)")
            variablesready = False

        if o is None or np.shape(o) != (2,n):
            if instructions: print ("destination values array, o, not initalized or not in the right shape (2xn)")
            variablesready = False

        if mass is None or np.shape(mass) != (n,):
            if instructions: print ("mass array not initialized or not with correct shape (n). It will be initailized with default values when initalizing global parameters - randomly uniform values between 60 and 100")

        if v_0 is None or np.shape(v_0) != (n,):
            if instructions: print ("comfortable walking speed array, v_0, not initialized or not with correct shape (n). It will be initailized with default values of 1.3m/s when initalizing global parameters")

        if v is None or np.shape(v) != (2,n):
            if instructions: print ("initial velocity array, v, not initialized or not with correct shape (2xn). It will be initailized with default values of zeros when initalizing global parameters")

        if n_walls is None:
            if instructions: print ("number of walls, n_walls, not initalized. It will be assumed to be 0 when initalizing global parameters")
        else:
            if walls is None or np.shape(walls) != (5,n_walls):
                if instructions: print ("numbers of walls initalized but array to store information about the walls not initialized or not with correct shape (5xn)")
                variablesready = False

    if variablesready:
        if instructions: print ("All necessary variables have been initalized. Call initialize_global_parameters() to initaize dependent parameters")
    else:
        if instructions: print ("Model is not ready. Please initialize required parameters")
#------------------------------------------------------------------------------#
def print_model_parameters():
    print ("tau = %4.2f, angular resolution in degrees = %4.2f, d_max = %4.2f, k = %4.2e, t = %4.2f" %( tau, math.degrees(ar), d_max, k, t ) )
#------------------------------------------------------------------------------#
def reset_model():
    """ Resets all inital conditions and sets model parameters to their default values """

    global variablesready, tau, ar, d_max, k, t, H_min, H_max, instructions, n, x, o, mass, v_0, v, n_walls, walls, color_p, time_step

    #Parameters of the model
    variablesready = False
    tau = 0.5 #second heurostic constant
    ar = math.radians(0.1) #angular resolution
    d_max = 10. #Horizon distance
    k = 5e3 #body collision constant
    t = 0 #Initial time set to 0
    H_min = math.radians(75)
    H_max = math.radians(75)
    instructions = False
    time_step = 0.05

    #Neccessary variables that that need to be initalized properly
    n = None #integer
    x = None #array of size 2xn
    o = None #array of size 2xn
    #Optional - default values initialized if not done so manually in func above
    mass = None #array of size n
    v_0 = None #array of size n
    v = None #array of size 2xn
    n_walls = None #integer
    walls = None #array of size 5xn - a,b,c,startwal, endwal
    #Optional - Not initalized if not specified as it has limited use
    color_p = None
#------------------------------------------------------------------------------#
def compute_destination_vals(i):
    """ Calculates v_des, f_alpha_des, and alpha_des values for given person i"""
    #Global values being saved
    global alpha_des, f_alpha_des, v_des

    result_i = compute_alpha_des(i)
    alpha_des[i] = result_i[0]
    f_alpha_des[i] = result_i[1]
    v_des[i] = min(v_0[i],f_alpha_des[i]/tau)
#------------------------------------------------------------------------------#
def compute_destinations():
    """ Calculates v_des, f_alpha_des, and alpha_des values for all persons"""
    for i in range(n):
        compute_destination_vals(i)
#------------------------------------------------------------------------------#
def move_pedestrians():
    """ Moves all pedestrians forward in time by time_step based on calculated v_des and alpha_des values"""
    #Global values being saved
    global v, x

    #acceleration due to body collisions - needs to computed before moving the pedestrians to ensure both people colliding feel the force
    abcx = np.zeros(n)
    abcy = np.zeros(n)
    for i  in range(n):
        [abcx[i],abcy[i]] = compute_bodycollision_acceleration(i)

    for i in range(n):
        #acceleration due to person adjusting position
        [a_x,a_y] = compute_iacceleration(v[0][i], v[1][i], v_des[i], alpha_des[i],tau)
        #acceleration due to body collisions
        ax_t = abcx[i] + a_x
        ay_t = abcy[i] + a_y
        v[0][i] = v[0][i] + ax_t * time_step
        v[1][i] = v[1][i] + ay_t * time_step
        x[0,i] = x[0,i] + time_step * v[0][i]
        x[1,i] = x[1,i] + time_step * v[1][i]
#------------------------------------------------------------------------------#
def update_model():
    """ Once alpha_des, v_des have been calculated and pedestrians have moved forward in time to ready the model for the next iteration"""

    global alpha_0, alpha_current, x_full, v_full, t
    #update alpha_0 values
    for i in range(n):
        alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
        if (v[0][i] == 0 and v[1][i] == 0):
            alpha_current[i] = alpha_0[i]
        else:
            alpha_current[i] = np.arctan2(v[1][i],v[0][i])
    #save information about positions of each individual
    x_full = np.dstack((x_full,x))
    v_full = np.dstack((v_full,v))
    #increment time
    t = t + time_step
#------------------------------------------------------------------------------#
def advance_model():
    """Advances current model in time by time_step"""

    compute_destinations()
    move_pedestrians()
    update_model()
#------------------------------------------------------------------------------#
def distances_to_point(xp,yp):
    """Computes the distances from a point (x,y) to all the pedestrians"""

    x_vals = x[0,:] - xp
    y_vals = x[1,:] - yp
    return np.linalg.norm( [x_vals,y_vals], axis = 0)
#------------------------------------------------------------------------------#
def weight_function(dist):
    """Computes the distance based weight function for an inputted value/array"""

    R = 0.7
    return np.exp( - np.multiply( dist, dist) / (R * R)) / ( np.pi * R * R)
#------------------------------------------------------------------------------#
def local_speed_current(point):
    """Computes the local speed at point x in the current iteraion of the model"""

    dist = distances_to_point(point[0],point[1])
    wf = weight_function( dist)
    sum_wf = np.sum(wf)

    if sum_wf == 0:
        return 0

    return np.sum( np.multiply( np.linalg.norm( v,axis = 0), wf)) / np.sum( wf)
#------------------------------------------------------------------------------#
def area_occupied():
    """Computes total area the pedestrians occupy - not accounting for overlaps"""

    # Sum over i of [ pi * r_i * r_i ]
    return pi * np.sum( np.multiply( r, r))
#------------------------------------------------------------------------------#
def occupancy(bounding_area):
    """Computes occupancy given bounding area - not accounting for overlaps"""

    return area_occupied() / bounding_area
#------------------------------------------------------------------------------#
def current_average_speed():
    """Computes the average speed in the current iteration of the model """

    return np.average( np.linalg.norm( v, axis = 0))
#------------------------------------------------------------------------------#
def average_speed():
    """Computes the average speed in the current iteration of the model """

    return np.average( np.linalg.norm( v_full, axis = 0))
#------------------------------------------------------------------------------#
#Initialize values of None for all variables
reset_model()


n_vals = [40,50,60,65,70,75,80] #number of people
# n_vals = [60]
t_iteration_vals = [None] * len(n_vals)
t_vals = [None] * len(n_vals)
occupancy_vals = [None] * len(n_vals)
avgspeed_vals = [None] * len(n_vals)

for index in range(len(n_vals)):
    #Set number of pedestrians
    n = n_vals[index]
    print ("\nn: ", n)

    #Parameters
    time_step = 0.05
    d_max = 8
    H_min = math.radians(45)
    H_max = math.radians(45)
    v_0 = np.random.normal( 1.3, 0.2, n)
    # model.v_0 = np.ones(model.n) * 1.3

    #Current position uniformly distributed between 0 and 100
    x = np.zeros( ( 2, n))
    spacing = 8. / ( n + 1)
    for i in range(n):
        x[0][i] = - 4. + ( i + 1.) * spacing
    x[1][::] = np.random.uniform( -1.15, 1.15, n)

    #Destination points, o, set to (50,0) for all
    o = np.zeros( ( 2, n))
    o[0,:] = 50.
    #Initialize the walls [a,b,c,startval,endval]
    n_walls = 2
    walls = np.zeros( (5, n_walls))
    # wall y = -1.5
    walls[:,0] = np.array([ 0, 1, 1.5, -4, 4])
    # wall y = 1.5
    walls[:,1] = np.array([ 0, 1, -1.5, -4, 4])

    check_model_ready()
    initialize_global_parameters()
    #------------------------------------------------------------------------------#
    #Increment the time
    start_time = time.clock()
    while (t<6):
        
        #compute alpha_des and v_des for each i
        compute_destinations()
        move_pedestrians()
        for i in range(n):
                if x[0,i] > 4:
                    x[0,i] = x[0,i] - 8
                if x[0,i] < -4:
                    x[0,i] = x[0,i] + 8
        #Update alpha_0 and alpha_current
        update_model()
        if t == time_step:
            t_iteration_vals[index] = time.clock() - start_time
            print( "Time for 1 iteration: %.3f" %(t_iteration_vals[index]) )
    end_time = time.clock()
    t_vals[index] = end_time - start_time
    print( "Time Taken: %.3f" %(t_vals[index]) )
    print(np.all(x_full[1]>-1.5) and np.all(x_full[1]<1.5))
    #------------------------------------------------------------------------------#
    bounding_area = 3. * 8.
    occupancy_vals[index] = occupancy(bounding_area)
    
    print ("Occupancy =", occupancy_vals[index])
    avgspeed_vals[index] = average_speed()
    print ("Average Speed = ", avgspeed_vals[index])
    print ("Comfortable Walking Speeds =", v_0)
    # fig1 = plt.figure()
    # model.plot_current_positions(fig1)
    # plt.axis([-4,4,-2,2])
    # plt.axhline(1.5,color = 'k', label = 'wall')
    # plt.axhline(-1.5,color = 'k', label = 'wall')
    # plt.savefig('testing.png')
    #------------------------------------------------------------------------------#
    reset_model()
#------------------------------------------------------------------------------#
print ("\nn:\n", n_vals)
print ("Occupancy:\n", occupancy_vals)
print ("Average Speed:\n", avgspeed_vals)
print ("Time Taken:\n", t_vals)
print ("Iteration Times:\n", t_iteration_vals)
#------------------------------------------------------------------------------#
np.savetxt('results_occupancy.out', (n_vals, occupancy_vals, avgspeed_vals, t_vals, t_iteration_vals), delimiter = ',')
#------------------------------------------------------------------------------#
# IMPORT DATA
data = np.genfromtxt('../data/data_pub_occupancy.csv', delimiter=',')
#------------------------------------------------------------------------------#
fig = plt.figure()
plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
plot_results = plt.plot(occupancy_vals,avgspeed_vals,'x-', label = 'My Results')
plt.legend(loc=1)
plt.title('Occupancy Comparison')
plt.xlabel('Occupancy')
plt.ylabel('Avg Speed (m/s)')
# plt.show()
plt.savefig('c_occupancy.png')