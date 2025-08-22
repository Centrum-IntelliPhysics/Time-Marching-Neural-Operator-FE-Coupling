import jax
import jax.numpy as jnp
from jax import grad, vmap, random, jit, config
import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.fem import functionspace 
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Constant, Function 
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from dolfinx.mesh import create_box, create_unit_square
import ufl 
from ufl import dx
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
import os 
from tqdm import trange 
import gmsh 
import math
from dolfinx import plot
from scipy.interpolate import Rbf, interp1d, griddata
import logging 
from matplotlib.ticker import ScalarFormatter
import time
from dynamic_utils import plot_disp

###############Attention##################
# the Dofinx 0.9.0 version is used in this code
# .vector --> .x.petsc_vec
##########################################


#### utils 
start_time = time.time()
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)


#region Save path       
originalDir = os.getcwd()
os.chdir(os.path.join(originalDir))

foldername = 'FE_full_elasto_dynamic_ground_truth'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

origin_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.02

# region gmsh
# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
radius1 = 0.3 - 0.05
num_points = 200      # Number of points on the circumference #so important !!!0 
num_points1 = 80  


# region full domain 
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
# Define inner square
# add the square points
# add the square points
gmsh.model.occ.addPoint(-1, -0.5, 0, lc, 1)
gmsh.model.occ.addPoint(1, -0.5, 0, lc, 2)
gmsh.model.occ.addPoint(1, 1.5, 0, lc, 3)
gmsh.model.occ.addPoint(-1, 1.5, 0, lc, 4)

# add the square lines
line1 = gmsh.model.occ.addLine(1, 2)
line2 = gmsh.model.occ.addLine(2, 3)
line3 = gmsh.model.occ.addLine(3, 4)
line4 = gmsh.model.occ.addLine(4, 1)

# create a loop from the lines
rec_loop = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])

# create a surface from the loop
background = gmsh.model.occ.addPlaneSurface([rec_loop])
#background = gmsh.model.occ.addRectangle(-1, -0.5, 0, 2, 2, 100)
gmsh.model.occ.synchronize()

# Define inner square
# add the square points
gmsh.model.occ.addPoint(center[0] - (radius + 0.05), center[1] - (radius + 0.05), 0, lc, 5)
gmsh.model.occ.addPoint(center[0] - (radius + 0.05), center[1] + (radius + 0.05), 0, lc, 6)
gmsh.model.occ.addPoint(center[0] + (radius + 0.05), center[1] + (radius + 0.05), 0, lc, 7)
gmsh.model.occ.addPoint(center[0] + (radius + 0.05), center[1] - (radius + 0.05), 0, lc, 8)
l_seq = 2*(radius + 0.05)/ (num_points1+1)

points_left = []
for i in range(num_points1):
    x_left = center[0] - (radius + 0.05)
    y_left = center[1] - (radius + 0.05) + (i+1)*l_seq
    points_left.append(gmsh.model.occ.addPoint(x_left, y_left, center[2], lc))

# Create lines between consecutive points
lines_left = []
for i in range(num_points1-1):
    p1_left = points_left[i]
    p2_left = points_left[i + 1]
    lines_left.append(gmsh.model.occ.addLine(p1_left, p2_left))

points_up = []
for i in range(num_points1):
    x_up = center[0] - (radius + 0.05) + (i+1)*l_seq
    y_up = center[1] + (radius + 0.05)
    points_up.append(gmsh.model.occ.addPoint(x_up, y_up, center[2], lc))

# Create lines between consecutive points
lines_up = []
for i in range(num_points1-1):
    p1_up = points_up[i]
    p2_up = points_up[i + 1]
    lines_up.append(gmsh.model.occ.addLine(p1_up, p2_up))

points_right = []
for i in range(num_points1):
    x_right = center[0] + (radius + 0.05)
    y_right = center[1] + (radius + 0.05) - (i+1)*l_seq
    points_right.append(gmsh.model.occ.addPoint(x_right, y_right, center[2], lc))

# Create lines between consecutive points
lines_right = []
for i in range(num_points1-1):
    p1_right = points_right[i]
    p2_right = points_right[i + 1]
    lines_right.append(gmsh.model.occ.addLine(p1_right, p2_right))

points_low = []
for i in range(num_points1):
    x_low = center[0] + (radius + 0.05) - (i+1)*l_seq
    y_low = center[1] - (radius + 0.05)
    points_low.append(gmsh.model.occ.addPoint(x_low, y_low, center[2], lc))

# Create lines between consecutive points
lines_low = []
for i in range(num_points1-1):
    p1_low = points_low[i]
    p2_low = points_low[i + 1]
    lines_low.append(gmsh.model.occ.addLine(p1_low, p2_low))


# add the square lines
line1_inner = gmsh.model.occ.addLine(5, points_left[0])
li1 = gmsh.model.occ.addLine(points_left[-1],6)
line2_inner = gmsh.model.occ.addLine(6, points_up[0])
li2 = gmsh.model.occ.addLine(points_up[-1], 7)
line3_inner = gmsh.model.occ.addLine(7, points_right[0])
li3 = gmsh.model.occ.addLine(points_right[-1], 8)
line4_inner = gmsh.model.occ.addLine(8, points_low[0])
li4 = gmsh.model.occ.addLine(points_low[-1], 5)

Boundary_left = [line1_inner] + lines_left + [li1]
Boundary_up = [line2_inner] + lines_up + [li2]
Boundary_right = [line3_inner] + lines_right + [li3]
Boundary_low = [line4_inner] + lines_low + [li4]

inner_lines = [line1_inner] + lines_left + [li1,line2_inner] + lines_up+ \
                                                [li2, line3_inner] + lines_right + [li3,line4_inner] + lines_low + [li4]
# create a loop from the lines
rec_loop_inner = gmsh.model.occ.addCurveLoop([line1_inner] + lines_left + [li1,line2_inner] + lines_up+ \
                                                [li2, line3_inner] + lines_right + [li3,line4_inner] + lines_low + [li4])
# create a surface from the loop
inner_square = gmsh.model.occ.addPlaneSurface([rec_loop_inner])
gmsh.model.occ.synchronize()

# Define the inner_in square
# add the square points
gmsh.model.occ.addPoint(center[0] - (radius), center[1] - (radius), 0, lc, 1115)
gmsh.model.occ.addPoint(center[0] - (radius), center[1] + (radius), 0, lc, 1116)
gmsh.model.occ.addPoint(center[0] + (radius), center[1] + (radius), 0, lc, 1117)
gmsh.model.occ.addPoint(center[0] + (radius), center[1] - (radius), 0, lc, 1118)

l_seq = 2*(radius)/ (num_points1+1)

points_left_in = []
for i in range(num_points1):
    x = center[0] - (radius)
    y = center[1] - (radius) + (i+1)*l_seq
    points_left_in.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_left_in = []
for i in range(num_points1-1):
    p1 = points_left_in[i]
    p2 = points_left_in[i + 1]
    lines_left_in.append(gmsh.model.occ.addLine(p1, p2))

points_up_in = []
for i in range(num_points1):
    x = center[0] - (radius) + (i+1)*l_seq
    y = center[1] + (radius)
    points_up_in.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_up_in = []
for i in range(num_points1-1):
    p1 = points_up_in[i]
    p2 = points_up_in[i + 1]
    lines_up_in.append(gmsh.model.occ.addLine(p1, p2))

points_right_in = []
for i in range(num_points1):
    x = center[0] + (radius)
    y = center[1] + (radius) - (i+1)*l_seq
    points_right_in.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_right_in = []
for i in range(num_points1-1):
    p1 = points_right_in[i]
    p2 = points_right_in[i + 1]
    lines_right_in.append(gmsh.model.occ.addLine(p1, p2))

points_low_in = []
for i in range(num_points1):
    x = center[0] + (radius) - (i+1)*l_seq
    y = center[1] - (radius)
    points_low_in.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_low_in = []
for i in range(num_points1-1):
    p1 = points_low_in[i]
    p2 = points_low_in[i + 1]
    lines_low_in.append(gmsh.model.occ.addLine(p1, p2))


# add the square lines
line1_inner_in = gmsh.model.occ.addLine(1115, points_left_in[0])
li1_in = gmsh.model.occ.addLine(points_left_in[-1],1116)
line2_inner_in = gmsh.model.occ.addLine(1116, points_up_in[0])
li2_in = gmsh.model.occ.addLine(points_up_in[-1], 1117)
line3_inner_in = gmsh.model.occ.addLine(1117, points_right_in[0])
li3_in = gmsh.model.occ.addLine(points_right_in[-1], 1118)
line4_inner_in = gmsh.model.occ.addLine(1118, points_low_in[0])
li4_in = gmsh.model.occ.addLine(points_low_in[-1], 1115)

inner_lines_in = [line1_inner_in] + lines_left_in + [li1_in, line2_inner_in] + lines_up_in+ \
                                                [li2_in, line3_inner_in] + lines_right_in + [li3_in,line4_inner_in] + lines_low_in + [li4_in]
# create a loop from the lines
rec_loop_inner_in = gmsh.model.occ.addCurveLoop(inner_lines_in)
# create a surface from the loop
inner_square_in = gmsh.model.occ.addPlaneSurface([rec_loop_inner_in])

# Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2,background),(2, inner_square), (2, inner_square_in)],  # Target entities
    []  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2) #2 --> 2d surface

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], tag=2)
inner_inner_region = gmsh.model.addPhysicalGroup(2, [surfaces[2][1]], tag=3)
#gmsh.model.addPhysicalGroup(2, [surface_hole], name=" Hole")


# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.setPhysicalName(2, inner_region, "Inner_Square")
gmsh.model.setPhysicalName(2, inner_inner_region, "Inner_Inner_Square")
gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4] , name="LargerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, inner_lines , name="InnerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, inner_lines_in, name="HoleEdges") # facet tag = 4

# Write the mesh to a file (optional)
gmsh.write("full_square.msh")

# Finalize GMSH
gmsh.finalize()


import meshio

# read (.msh)
msh = meshio.read("full_square.msh")
# extract points and cells
points = msh.points
cells = msh.cells_dict["triangle"]  
plt.figure(figsize=(8, 8))
# draw the mesh
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5)  # 用黑色线条绘制
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization')
plt.savefig('full_square' + ".jpg", dpi=700)
plt.show()

# region FEM
# read .msh
mesh, cell_markers, facet_markers = gmshio.read_from_msh("full_square.msh", MPI.COMM_WORLD)
V = functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))  ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space
tdim = mesh.topology.dim  #mesh.geometry.dim = 3 mesh.topology.dim = 2
fdim = tdim - 1 # facet dimension
# extract the coordinates of the mesh (each vertice of the mesh)
X, Y= mesh.geometry.x[:,0], mesh.geometry.x[:,1]
np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)

# find the inner square points and index 
index_inner_square = np.where(
    ((X < center[0] + radius + 0.05 + lc*1e-4) & (X > center[0] - radius - 0.05 - lc*1e-4)) &
    ((Y < center[1] + radius + 0.05 + lc*1e-4) & (Y > center[1] - radius - 0.05 - lc*1e-4))
)[0]

#print('index_inner_sqaure:', index_inner_square.shape)
plot_disp(X[index_inner_square], Y[index_inner_square], X[index_inner_square], 'inner_square', 'Inner Square')
X1 = X[index_inner_square]
Y1 = Y[index_inner_square]

index_up = np.where(np.isclose(Y1, center[1] + (radius + 0.05)))[0]
index_down = np.where(np.isclose(Y1, center[1] - (radius + 0.05)))[0]
index_left = np.where(np.isclose(X1, center[0] - (radius + 0.05)))[0]
index_right = np.where(np.isclose(X1, center[0] + (radius + 0.05)))[0]
print('index_up:', index_up.shape)
print('index_down:', index_down.shape)
print('index_left:', index_left.shape)
print('index_right:', index_right.shape)

np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma_(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  
# Sub domain for clamp at bottom 
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)
# Sub domain for rotation at top
def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)

# region Parameters
# steel params 
E  = 1000 #210e9 #Pa
nu = 0.3
mu    =  E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density
rho = 5 #7800 #kg/m^3

# Newmark method parameters
gamma   =  0.5 #Constant(mesh, 0.5+alpha_f-alpha_m)
beta    =  1 #Constant(mesh, (gamma+0.5)**2/4.)

# Time-stepping parameters
T       = 4.0
Nsteps  = 1e4
dt =  T/Nsteps

# the p-wave for elasto-dynamic problem
c_p =((lmbda + 2*mu)/rho)**0.5

print('Obey the Courant-Friedrichs-Lewy (CFL) condition :', dt < lc/c_p)


# Test and trial functions
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)
domain = mesh 


# region Elastic funcs
# Stress tensor
def sigma(r):
    return 2.0*mu*ufl.sym(ufl.grad(r)) + lmbda*ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r))

# Mass form
def m(u, u_):
    return rho*ufl.inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*dx

# Rayleigh damping form
'''def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)'''

# a = 2/((u - u0 - v0*dt)/(beta*dt*dt) - (1-beta)*a0)
# region update formula
# Update formula for acceleration
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
        # transform vector to array
        u_val = u.array
        u_old_val = u_old.array
        v_old_val = v_old.array
        a_old_val = a_old.array
        
        # update a value 
        a_new_val = 2*(u_val - u_old_val - dt_ * v_old_val) / (beta_ * dt_**2) - \
                     (1 -  beta_) / beta_* a_old_val
                  
        # back to vector 
        a_old.setArray(a_new_val)
        a_old.assemble()  # assemble vector 
        
    return a_old 

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
        
        # transform vector to array
        a_val = a.array
        u_old_val = u_old.array
        v_old_val = v_old.array
        a_old_val = a_old.array
        
        # update a value 
        v_new_val = v_old_val + dt_*((1-gamma_)*a_old_val + gamma_*a_val)
        
        # back to vector 
        v_old.setArray(v_new_val)
        v_old.assemble()  # assemble vector 
        
    return v_old 

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step.""" 

    # Get vectors (references)
    u_vec, u0_vec  = u.x.petsc_vec, u_old.x.petsc_vec
    v0_vec, a0_vec = v_old.x.petsc_vec, a_old.x.petsc_vec 

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.x.petsc_vec[:], a_old.x.petsc_vec[:] = v_vec, a_vec
    u_old.x.petsc_vec[:] = u.x.petsc_vec


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
ts_end = 120

break_list=[]
u_list, v_list, u2_list, v2_list = [], [], [], []

t=0  # initial time step 
# Test and trial functions
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# region BCs
# Set up boundary condition at bottom
bt_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bt)
uD = np.array([0, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_b)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

bt_l = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left)
uD_l = np.array([0, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_l)
bc_l = fem.dirichletbc(uD_l, boundary_dofs, V)


# Set up boundary condition at top 
top_b = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)
uD_top = np.array([0, 0.01, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)

top_r = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right)
uD_r = np.array([0.01, 0, 0], dtype=default_scalar_type)
mesh.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, top_r)
bc_r = fem.dirichletbc(uD_r, boundary_dofs, V)

bc = [bc_bt, bc_l, bc_top, bc_r]

# region [M][a] = [F]
# find the inner square points and index 
index_inner_square = np.where(
    ((X < center[0] + radius + 0.05 + lc*1e-4) & (X > center[0] - radius - 0.05 - lc*1e-4)) &
    ((Y < center[1] + radius + 0.05 + lc*1e-4) & (Y > center[1] - radius - 0.05 - lc*1e-4))
)[0]

#print('index_inner_sqaure:', index_inner_square.shape)
plot_disp(X[index_inner_square], Y[index_inner_square], X[index_inner_square], 'inner_square', 'Inner Square')
X1 = X[index_inner_square]
Y1 = Y[index_inner_square] 

t=0  # initial time step 
#for i in range(Nsteps):
break_list=[]
for ts in trange(141):
    t += dt    
    error_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    # Newmark time-stepping scheme
    LL = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
    RR = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + (1 -  beta) / beta* a_old , u_)*dx 

    ## FEM 
    # assemble the matrix and vector
    a_form = fem.form(LL)
    L_form = fem.form(RR)
    # Define solver (Not for reuse)
    A = assemble_matrix(a_form, bcs=bc)
    A.assemble()
    b = create_vector(L_form)
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L_form)
    
    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a_form], [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bc)
    # Solve linear problem
    solver.solve(b, u.x.petsc_vec)
    u.x.scatter_forward()

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_values = u.x.array.real
    update_fields(u, u_old, v_old, a_old)
    u_tot = u_values.reshape(-1,3)
    U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]
    X, Y = u_geometry[:,0], u_geometry[:,1]

    # retrieve stress 
    gdim = mesh.geometry.dim
    ### stress is 3*3  (gdim, gdim)
    Function_space_for_sigma = fem.functionspace(mesh, ("CG", 1, (gdim, gdim)))        
    expr= fem.Expression(sigma(u), Function_space_for_sigma.element.interpolation_points())        
    sigma_values = Function(Function_space_for_sigma) 
    sigma_values.interpolate(expr)
    sigma_tot = sigma_values.x.array.reshape(-1,9)
    
    '''if ts >= 89 and ts <= 140:
        U_1 = U_[index_inner_square]
        V_1 = V_[index_inner_square]
        ax = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_square, 0]
        ay = a_old.x.petsc_vec.array.reshape(-1,3)[index_inner_square, 1]
        vx = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_square, 0]
        vy = v_old.x.petsc_vec.array.reshape(-1,3)[index_inner_square, 1]
        sigma_x1 = sigma_tot[index_inner_square,0]
        sigma_y1 = sigma_tot[index_inner_square,4]
        sigma_xy1 = sigma_tot[index_inner_square,1]
        np.savetxt('U ts = ' + str(ts) +'.txt', U_)
        np.savetxt('V ts = ' + str(ts) +'.txt', V_)
        np.savetxt('U1 ts = ' + str(ts) +'.txt', U_1)
        np.savetxt('V1 ts = ' + str(ts) +'.txt', V_1)
        np.savetxt('ax ts = ' + str(ts) +'.txt', ax)
        np.savetxt('ay ts = ' + str(ts) +'.txt', ay)
        np.savetxt('vx ts = ' + str(ts) +'.txt', vx)
        np.savetxt('vy ts = ' + str(ts) +'.txt', vy)
        np.savetxt('sigma_x1 ts = ' + str(ts) +'.txt', sigma_x1)
        np.savetxt('sigma_y1 ts = ' + str(ts) +'.txt', sigma_y1)
        np.savetxt('sigma_xy1 ts = ' + str(ts) +'.txt', sigma_xy1)'''


    if ts == ts_end or ts == ts_end + 10 or ts == ts_end + 20:
        plot_disp(X,Y,U_, 'displacement u ts=' + str(ts), rf'$u_{{x, \mathrm{{FE}}}}^{{{ts}}}$')
        plot_disp(X,Y,V_,'displacement v ts=' + str(ts), rf'$u_{{y, \mathrm{{FE}}}}^{{{ts}}}$')
        #plot_disp(X,Y,sigma_tot[:,0], 'sigma_x ts=' + str(ts), rf'$\sigma_{{x,\mathrm{{FE}}}}^{{{ts}}}$')
        #plot_disp(X,Y,sigma_tot[:,4],'sigma_y ts=' + str(ts), rf'$\sigma_{{y,\mathrm{{FE}}}}^{{{ts}}}$')
        # save the data for interpolation to calculate the error 
        np.savetxt('X.txt', X)
        np.savetxt('Y.txt', Y)
        np.savetxt('u ts=' + str(ts) + '.txt', U_)
        np.savetxt('v ts=' + str(ts) + '.txt', V_)
        np.savetxt('sigma_x ts=' + str(ts) + '.txt', sigma_tot[:,0])
        np.savetxt('sigma_y ts=' + str(ts) + '.txt', sigma_tot[:,4])

end_time= time.time()
print('Time cost:', end_time - start_time)









        




