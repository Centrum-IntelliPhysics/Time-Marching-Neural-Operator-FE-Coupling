'''
====================================================================
FE-DeepONet_coupling for elasto-dynamic problem with square domain
====================================================================

'''

import jax
import jax.numpy as jnp
from jax import grad, vmap
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
import pickle
from jax import random
from matplotlib.ticker import ScalarFormatter
from dynamic_utils import plot_disp, plot_disp_real, plot_error_list, plot_mesh, plot_boundary

###############Attention##################
# the Dofinx 0.9.0 version is used in this code
# .vector --> .x.petsc_vec
##########################################


# region Save Path 
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)
 
originalDir = os.getcwd()
os.chdir(os.path.join(originalDir))

foldername = 'FE_DeepONet_coupling_elasto_dynamic_suqare_square_89_139_vmax_vmin'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
 
originalDir_real = os.path.join(originalDir, './'+ foldername + '/')



# Show the plot
plt.show()

#region square_hole 
lc = 0.02

# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
num_points = 200   # Number of points on the circumference #so important !!!0 
num_points1 = 80
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
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

# cut the inner_in square 
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
gmsh.model.occ.synchronize()


# Cut the larger square with the hole
gmsh.model.occ.cut([(2, background)], [(2, inner_square_in)])
# Synchronize the model to GMSH
gmsh.model.occ.synchronize()
# insert the inner square into the background
# Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2, background)],  # Target entities
    [(2, inner_square)]  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()
# Generate 2D mesh

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [background], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [inner_square], tag=2)

gmsh.model.mesh.generate(2)

gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4], name="LargerSquareEdges") # facet tag = 7
gmsh.model.addPhysicalGroup(1, inner_lines , name="InnerSquareEdges") # facet tag = 8 interupt the boundary 
gmsh.model.addPhysicalGroup(1, inner_lines_in, name="smallsqaureEdges")# facet tag = 9

# Write the mesh to a file (optional)
gmsh.write("outer_domain.msh")

# Finalize GMSH
gmsh.finalize()


# region Hole 
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
# Define inner square
# add the square points
gmsh.model.occ.addPoint(center[0] - (radius + 0.05), center[1] - (radius + 0.05), 0, lc, 5)
gmsh.model.occ.addPoint(center[0] - (radius + 0.05), center[1] + (radius + 0.05), 0, lc, 6)
gmsh.model.occ.addPoint(center[0] + (radius + 0.05), center[1] + (radius + 0.05), 0, lc, 7)
gmsh.model.occ.addPoint(center[0] + (radius + 0.05), center[1] - (radius + 0.05), 0, lc, 8)

l_seq = 2*(radius + 0.05)/ (num_points1+1)

points_left = []
for i in range(num_points1):
    x = center[0] - (radius + 0.05)
    y = center[1] - (radius + 0.05) + (i+1)*l_seq
    points_left.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_left = []
for i in range(num_points1-1):
    p1 = points_left[i]
    p2 = points_left[i + 1]
    lines_left.append(gmsh.model.occ.addLine(p1, p2))

points_up = []
for i in range(num_points1):
    x = center[0] - (radius + 0.05) + (i+1)*l_seq
    y = center[1] + (radius + 0.05)
    points_up.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_up = []
for i in range(num_points1-1):
    p1 = points_up[i]
    p2 = points_up[i + 1]
    lines_up.append(gmsh.model.occ.addLine(p1, p2))

points_right = []
for i in range(num_points1):
    x = center[0] + (radius + 0.05)
    y = center[1] + (radius + 0.05) - (i+1)*l_seq
    points_right.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_right = []
for i in range(num_points1-1):
    p1 = points_right[i]
    p2 = points_right[i + 1]
    lines_right.append(gmsh.model.occ.addLine(p1, p2))

points_low = []
for i in range(num_points1):
    x = center[0] + (radius + 0.05) - (i+1)*l_seq
    y = center[1] - (radius + 0.05)
    points_low.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines_low = []
for i in range(num_points1-1):
    p1 = points_low[i]
    p2 = points_low[i + 1]
    lines_low.append(gmsh.model.occ.addLine(p1, p2))


# add the square lines
line1_inner = gmsh.model.occ.addLine(5, points_left[0])
li1 = gmsh.model.occ.addLine(points_left[-1],6)
line2_inner = gmsh.model.occ.addLine(6, points_up[0])
li2 = gmsh.model.occ.addLine(points_up[-1], 7)
line3_inner = gmsh.model.occ.addLine(7, points_right[0])
li3 = gmsh.model.occ.addLine(points_right[-1], 8)
line4_inner = gmsh.model.occ.addLine(8, points_low[0])
li4 = gmsh.model.occ.addLine(points_low[-1], 5)

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
gmsh.model.occ.synchronize()

    # Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2, inner_square)],  # Target entities
    [(2, inner_square_in)]  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], tag=2)
#gmsh.model.addPhysicalGroup(2, [surface_hole], name=" Hole")


# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
gmsh.model.setPhysicalName(2, inner_region, "Inner_Square")
gmsh.model.addPhysicalGroup(1, inner_lines , name="InnerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, inner_lines_in, name="InnerinsquareEdges") # facet tag = 4
gmsh.write("inner_hole.msh")

# Finalize GMSH
gmsh.finalize()

import meshio

# region Plot the mesh
msh = meshio.read("outer_domain.msh")
# extract points and cells information
points = msh.points
cells = msh.cells_dict["triangle"]  
# create figure
plot_mesh(cells, points, 'outer_domain')


# read the mesh file
msh = meshio.read("inner_hole.msh")
# extract points and cells information
points = msh.points
cells = msh.cells_dict["triangle"]  
# create figure
plot_mesh(cells, points, 'inner_domain')


# region FEM 
# import .msh 
mesh, cell_markers, facet_markers  = gmshio.read_from_msh("outer_domain.msh", MPI.COMM_WORLD) 
V = functionspace(mesh, ("CG", 1, (mesh.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
#mesh = create_unit_square(MPI.COMM_WORLD, 100, 100) 
tdim = mesh.topology.dim  #mesh.geometry.dim = 3 mesh.topology.dim = 2
fdim = tdim - 1 # facet dimension
# extract the coordinates of the mesh (every vertices of the mesh)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
X, Y = u_geometry[:,0], u_geometry[:,1]



def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma_(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  

# region locators
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



def low_inner(x):
    return np.isclose(x[1], center[1] - (radius + 0.05))

def left_inner(x):
    return np.isclose(x[0], center[0] - (radius + 0.05))
# Sub domain for rotation at top
def top_inner(x):
    return np.isclose(x[1], center[1] + (radius + 0.05)) 

def right_inner(x):
    return np.isclose(x[0], center[0] + (radius + 0.05))

def low_inner_in(x):
    return np.isclose(x[1], center[1] - (radius))

def left_inner_in(x):
    return np.isclose(x[0], center[0] - (radius))
# Sub domain for rotation at top
def top_inner_in(x):
    return np.isclose(x[1], center[1] + (radius)) 

def right_inner_in(x):
    return np.isclose(x[0], center[0] + (radius))



# region Parameters
# steel params 
E  = 1000 
nu = 0.3
mu    =  E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density
rho = 5 
# Newmark method parameters
gamma   =  0.5
beta    =  1 

# Time-stepping parameters
T       = 4.0
Nsteps  = 1e4
dt =  T/Nsteps

# the p-wave for elasto-dynamic problem
c_p =((lmbda + 2*mu)/rho)**0.5

print('Obey the Courant-Friedrichs-Lewy (CFL) condition :', dt < lc/c_p)

# External pressure
p0 = 1.
cutoff_Tc = T/5

# Define function space for stresses
Vsig = fem.functionspace(mesh, ("DG", 0, (tdim, tdim)))

# Test and trial functions
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)
p = fem.Function(V)


#region Exterior loadings  
### not use in this code ###
def pressure_expression(t):
    if t <= cutoff_Tc:
        return p0 * t / cutoff_Tc
    else:
        return p0

def update_pressure(t):
    p_array = p.vector.getArray()
    p_array_x = p_array.reshape(-1,2)[:,0]
    p_array_y = p_array.reshape(-1,2)[:,1]
    for i in range(p_array_x.size):
        p_array_x[i] = 0.
    for i in range(p_array_y.size):
        p_array_y[i] = pressure_expression(t)    
    p_array_new = np.hstack([p_array_x.reshape(-1,1),p_array_y.reshape(-1,1)])
    #print(p_array_new.shape)
    p.vector.setArray(p_array_new.flatten())

# top displacement 
def top_disp(t):
    return 0.01 * t

# region Measure dss
# Create mesh function over the cell facets
domain = mesh 
# locate the boundary facets
d_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, top)
# Mark the facets
d_facets_mark = np.zeros_like(d_facets) + 3
# Create mesh tag (assign a number to the specific facets)
ft = dolfinx.mesh.meshtags(mesh, fdim, np.array(d_facets).astype(np.int32), 
                   np.array(d_facets_mark).astype(np.int32))
mesh.topology.create_connectivity(fdim, tdim)
# Define measure for boundary condition integral
dss = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
#To verify the inner surface is well-difined 
facet_values = ft.values
facet_indices = ft.indices
# Define a color map for the tags
colors = {3: 'red'}  # Assign colors for each tag (1, 2, 3)
### Attention fdim = 1, tdim = 2 ####
plot_boundary(mesh, facet_values, facet_indices, colors, 'top_boundary', fdim)


# region boundary ptx
def on_inner_top(x):
    '''
    return the coordinates on the inner top 
    '''
    xc = x[0][np.where(np.isclose(x[1], center[1] + (radius + 0.05), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[1], center[1] + (radius + 0.05), 1e-6, 1e-6))]
    return xc, yc

def on_inner_low(x):
    '''
    return the coordinates on the inner low 
    '''
    xc = x[0][np.where(np.isclose(x[1], center[1] - (radius + 0.05), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[1], center[1] - (radius + 0.05), 1e-6, 1e-6))]
    return xc, yc

def on_inner_left(x):
    '''
    return the coordinates on the inner left 
    '''
    xc = x[0][np.where(np.isclose(x[0], center[0] - (radius + 0.05), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[0], center[0] - (radius + 0.05), 1e-6, 1e-6))]
    return xc, yc

def on_inner_right(x):
    '''
    return the coordinates on the inner right 
    '''
    xc = x[0][np.where(np.isclose(x[0], center[0] + (radius + 0.05), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[0], center[0] + (radius + 0.05), 1e-6, 1e-6))]
    return xc, yc




def on_inner_top_in(x):
    '''
    return the coordinates on the inner top 
    '''
    xc = x[0][np.where(np.isclose(x[1], center[1] + (radius), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[1], center[1] + (radius), 1e-6, 1e-6))]
    return xc, yc

def on_inner_low_in(x):
    '''
    return the coordinates on the inner low 
    '''
    xc = x[0][np.where(np.isclose(x[1], center[1] - (radius), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[1], center[1] - (radius), 1e-6, 1e-6))]
    return xc, yc

def on_inner_left_in(x):
    '''
    return the coordinates on the inner left 
    '''
    xc = x[0][np.where(np.isclose(x[0], center[0] - (radius), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[0], center[0] - (radius), 1e-6, 1e-6))]
    return xc, yc

def on_inner_right_in(x):
    '''
    return the coordinates on the inner right 
    '''
    xc = x[0][np.where(np.isclose(x[0], center[0] + (radius), 1e-6, 1e-6))]
    yc = x[1][np.where(np.isclose(x[0], center[0] + (radius), 1e-6, 1e-6))]
    return xc, yc



# region Expression 
'''
-------------------------------------------------------
expression for the interpolation of the boundary values
-------------------------------------------------------     
'''

class MyExpression_top:
    '''
    The class for interpolation in FenicsX
    ======================================
    Outer domain at top edge  
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], center[1] + (radius + 0.05), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], center[1] + (radius + 0.05), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 
    
class MyExpression_low:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Outer domain at bottom edge  
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], center[1] - (radius + 0.05), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], center[1] - (radius + 0.05), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 


class MyExpression_left:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Outer domain at left edge  
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[0], center[0] - (radius + 0.05), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[0], center[0] - (radius + 0.05), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values

class MyExpression_right:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Outer domain at right edge  
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[0], center[0] + (radius + 0.05), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[0], center[0] + (radius + 0.05), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values



class MyExpression_top_in:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Inner domain at top edge
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], center[1] + (radius), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], center[1] + (radius), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 
    
class MyExpression_low_in:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Inner domain at bottom edge
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[1], center[1] - (radius), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[1], center[1] - (radius), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 


class MyExpression_left_in:
    '''
    The class for interpolation in FenicsX 
    ======================================
    Inner domain at left edge
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[0], center[0] - (radius), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[0], center[0] - (radius), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values

class MyExpression_right_in:
    '''
    The class for interpolation in FenicsX
    ======================================
    Inner domain at right edge
    '''
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        #(x[0], x[1], x[2]) are (x, y, z), 
        # values[0], values[1], values[2] are corresponding values at points (x, y, z)
        # values are interpolated by RBF 
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose(x[0], center[0] + (radius), 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose(x[0], center[0] + (radius), 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values
    

# region Elastic funcs
# Stress tensor
def sigma(r):
    return 2.0*mu*ufl.sym(ufl.grad(r)) + lmbda*ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r))

# Mass form
def mass(u, u_):
    return rho*ufl.inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*dx

# Rayleigh damping form
'''def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)'''

# Work of external forces
def Wext(u_):
    return ufl.dot(u_, p)*dss(3)

# Update formula for acceleration
# a = 2/((u - u0 - v0*dt)/(beta*dt*dt) - (1-beta)*a0)
# region update formula
def update_a(u, u_old, v_old, a_old, ufl=True):
    #print('dt=', type(dt))
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


# update formula for NN 
def update_fields_NN(u, u_old, v_old, a_old):
    """
    Update fields at the end of each time step.
    a = 2/((u - u0 - v0*dt)/(beta*dt*dt) - (1-beta)*a0)
    v = dt * ((1-gamma)*a0 + gamma*a) + v0
    """ 
    a = 2*(u - u_old - dt * v_old) / (beta * dt**2) - \
                    (1 -  beta) / beta* a_old
    
    v = v_old + dt*((1-gamma)*a_old + gamma*a)

    # Update (u_old <- u)
    v_old[:], a_old[:] = v, a
    u_old[:] = u


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new


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


# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# region [M][a] = [F]
# Confusing bug: ufl.rhs and ufl.lhs cannot find the bilinear and linear forms correctly
# especially, when the corresponding fenicsx functions are used in elastic funcs
LL = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
RR = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + (1 -  beta) / beta* a_old , u_)*dx  
 

# region Iteration part 
### later these parameters will be replaced by DeepONet 
mesh2, cell_markers, facet_markers  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
V2 = functionspace(mesh2, ("CG", 1, (mesh2.geometry.dim, )))
du2 = ufl.TrialFunction(V2)
u_2 = ufl.TestFunction(V2)
# Current (unknown) displacement
u2 = Function(V2, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old2 = Function(V2)
v_old2 = Function(V2)
a_old2 = Function(V2)

# region DeepONet 
from prepare_DeepONet_Elasto_dynamic_square_square_99_109 import PI_DeepONet


m = num_points1 + 2
d = 2 # dimension of the input data
ela_model = dict()
ela_model['E'] = 1000e-8 #1000 
ela_model['nu'] = 0.3 
ela_model['rho'] = 5e-8 #5

branch_layers_1 =  [2*4*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers_1, trunk_layers, **ela_model)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_Elasto_dynamic_square_square_89_99' + '/'))
print(os.getcwd())
with open('DeepONet_ED_89_99.pkl', 'rb') as f:
    params_1 = pickle.load(f)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_Elasto_dynamic_square_square_99_109' + '/'))
print(os.getcwd())
with open('DeepONet_ED_99_109.pkl', 'rb') as f:
    params0 = pickle.load(f)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_Elasto_dynamic_square_square_109_119' + '/'))
print(os.getcwd())
with open('DeepONet_ED_109_119.pkl', 'rb') as f:
    params1 = pickle.load(f)


os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_Elasto_dynamic_square_square_119_129' + '/'))
print(os.getcwd())
with open('DeepONet_ED_119_129.pkl', 'rb') as f:
    params2 = pickle.load(f)

os.chdir(os.path.join(originalDir, './' + 'prepare_DeepONet_Elasto_dynamic_square_square_129_139' + '/'))
print(os.getcwd())
with open('DeepONet_ED_129_139.pkl', 'rb') as f:
    params3 = pickle.load(f)




os.chdir(originalDir_real)



t=0  # initial time step 
ts_end = 90 # final time step for FE_FE coupling
params = params_1 # initialize the parameters for DeepONet 
iter_NN =  0  # count the iteration of NN

break_list=[]

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V2)
X1, Y1 = u_geometry[:,0], u_geometry[:,1]
# predict the displacement for the whole domain
y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
'''
===========================================================
The handshake between subdomains is information exchange
at the overlapping boundaries
-----------------------------------------------------------
The index and coordinates of the boundary points extraction
Every domain has inner and outer edges in overlapping region 
===========================================================
'''
# coordinates for the inner domain inner edges
x_up_in, y_up_in = on_inner_top_in(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))
x_low_in, y_low_in = on_inner_low_in(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))
x_left_in, y_left_in = on_inner_left_in(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))
x_right_in, y_right_in = on_inner_right_in(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))

# index of the inner domain inner edges
coor_up_in  = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0]
                    for x_val, y_val in zip(x_up_in, y_up_in)])[:,0]
coor_low_in = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0]
                    for x_val, y_val in zip(x_low_in, y_low_in)])[:,0]
coor_left_in = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0]
                    for x_val, y_val in zip(x_left_in, y_left_in)])[:,0]
coor_right_in = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0]
                    for x_val, y_val in zip(x_right_in, y_right_in)])[:,0]


# index of the outer domain inner edges
index_up_out = np.where(np.isclose(Y, center[1] + (radius + 0.05), 1e-6))[0]
index_down_out = np.where(np.isclose(Y, center[1] - (radius + 0.05), 1e-6))[0]
index_left_out = np.where(np.isclose(X, center[0] - (radius + 0.05), 1e-6))[0]
index_right_out = np.where(np.isclose(X, center[0] + (radius + 0.05), 1e-6))[0]

# index of the inner domain outer edges 
index_up = np.where(np.isclose(Y1, center[1] + (radius + 0.05)))[0]
index_down = np.where(np.isclose(Y1, center[1] - (radius + 0.05)))[0]
index_left = np.where(np.isclose(X1, center[0] - (radius + 0.05)))[0]
index_right = np.where(np.isclose(X1, center[0] + (radius + 0.05)))[0]

remix_down = np.argsort(X1[index_down])
remix_up = np.argsort(X1[index_up])
remix_left = np.argsort(Y1[index_left])
remix_right = np.argsort(Y1[index_right])

'''
============================================================================
Attention: the order of the boundary points is important,
it should be consistent with the sequence of the points in DeepONet training
=============================================================================
'''
index_up = index_up[remix_up]
index_down = index_down[remix_down]
index_left = index_left[remix_left]
index_right = index_right[remix_right]

# coordinates for the outer domain outer edges 
x_up, y_up = on_inner_top(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_low, y_low = on_inner_low(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_left, y_left = on_inner_left(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_right, y_right = on_inner_right(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

#index of the outer domain outer edges
coor_up  = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_up, y_up)])[:,0]
coor_low = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_low, y_low)])[:,0]
coor_left = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_left, y_left)])[:,0]
coor_right = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_right, y_right)])[:,0]

# coordinates for the outer domain inner edges
x_up_in0, y_up_in0 = on_inner_top_in(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_low_in0, y_low_in0 = on_inner_low_in(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_left_in0, y_left_in0 = on_inner_left_in(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_right_in0, y_right_in0 = on_inner_right_in(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

# index of the outer domain inner edges
coor_up_in0  = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_up_in0, y_up_in0)])[:,0]
coor_low_in0 = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_low_in0, y_low_in0)])[:,0]
coor_left_in0 = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_left_in0, y_left_in0)])[:,0]
coor_right_in0 = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_right_in0, y_right_in0)])[:,0]


# coordiantes for Rbf to get structured grids (as CNN input)
X11 = np.linspace(center[0]-radius-0.05, center[0]+radius+0.05, m)
Y11 = np.linspace(center[1]-radius-0.05, center[1]+radius+0.05, m)
X1_, Y1_ = np.meshgrid(X11, Y11)


np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)
np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y) 

  

for ts in trange(140):
    t += dt    
    #update_pressure(t-dt)

    error_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    niter = 1000
    theta = 0.5 # relaxation coefficient

    #### Coupling  
    for iter in range(0, niter):
        #region First FE solver 
        if ts == 0 and iter == 0:
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
            
        
        if  (ts == 0 and iter > 0) or (ts > 0 and iter >= 0): 
            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            u_old.x.petsc_vec.setArray(u_old_0)
            v_old.x.petsc_vec.setArray(v_old_0)
            a_old.x.petsc_vec.setArray(a_old_0)
            
            # region Handshake 
            LL3 = rho*ufl.inner(2*(du) / (beta * dt**2), u_)*dx + \
                      ufl.inner(sigma(du), ufl.sym(ufl.grad(u_)))*dx     
            
            RR3 = rho*ufl.inner(2*( u_old + dt * v_old) / (beta * dt**2) + \
                                (1 -  beta) / beta* a_old , u_)*dx \
                                #+ ufl.dot(T_exp_c, u_)* ds_fem(1)

            a_form3 = fem.form(LL3)
            L_form3 = fem.form(RR3)
            A3 = assemble_matrix(a_form3, bcs=bc)
            A3.assemble()
            b3 = create_vector(L_form3)

            solver = PETSc.KSP().create(mesh.comm)
            solver.setOperators(A3)
            solver.setType(PETSc.KSP.Type.CG)
            solver.getPC().setType(PETSc.PC.Type.JACOBI)

            # Update the right hand side reusing the initial vector
            with b3.localForm() as loc_b3:
                loc_b3.set(0)
            assemble_vector(b3, L_form3)
            
            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b3, [a_form3], [bc])
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b3, bc)
            # Solve linear problem
            solver.solve(b3, u.x.petsc_vec)
            u.x.scatter_forward()

            u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
            u_values = u.x.array.real
        
        # Store the current values of u_old, v_old, a_old for the next iteration
        u_old_0 = np.copy(u_old.x.petsc_vec.array)
        v_old_0 = np.copy(v_old.x.petsc_vec.array)
        a_old_0 = np.copy(a_old.x.petsc_vec.array)

        # update the u, u_old, v_old, a_old(+ delta_t) 
        # Attention: we use stored values u_old_0, v_old_0, a_old_0 in one inner iteration
        update_fields(u, u_old, v_old, a_old)
        
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]


        
        # displacement on the four boundaries 
        u_up = U_[coor_up].reshape(1,-1)
        v_up = V_[coor_up].reshape(1,-1)

        u_low = U_[coor_low].reshape(1,-1)
        v_low = V_[coor_low].reshape(1,-1)

        u_left = U_[coor_left].reshape(1,-1)
        v_left = V_[coor_left].reshape(1,-1)

        u_right = U_[coor_right].reshape(1,-1)
        v_right = V_[coor_right].reshape(1,-1)

        
        # displacement on the four boundaries 
        u_up_in0 = U_[coor_up_in0].reshape(1,-1)
        v_up_in0 = V_[coor_up_in0].reshape(1,-1)

        u_low_in0 = U_[coor_low_in0].reshape(1,-1)
        v_low_in0 = V_[coor_low_in0].reshape(1,-1)

        u_left_in0 = U_[coor_left_in0].reshape(1,-1)
        v_left_in0 = V_[coor_left_in0].reshape(1,-1)

        u_right_in0 = U_[coor_right_in0].reshape(1,-1)
        v_right_in0 = V_[coor_right_in0].reshape(1,-1)
    
        # retrieve stress 
        gdim = mesh.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_sigma = fem.functionspace(mesh, ("CG", 1, (gdim, gdim)))        
        expr= fem.Expression(sigma(u), Function_space_for_sigma.element.interpolation_points())        
        sigma_values = Function(Function_space_for_sigma) 
        sigma_values.interpolate(expr)
        sigma_tot = sigma_values.x.array.reshape(-1,9)

        
        # region Inner Hole
        fdim2 = mesh2.topology.dim -1 
        up_line = dolfinx.mesh.locate_entities_boundary(mesh2, fdim2, top_inner)
        low_line = dolfinx.mesh.locate_entities_boundary(mesh2, fdim2, low_inner)
        left_line = dolfinx.mesh.locate_entities_boundary(mesh2, fdim2, left_inner)
        right_line = dolfinx.mesh.locate_entities_boundary(mesh2, fdim2, right_inner)

        # Set up boundary condition at top
        uD_up = Function(V2)
        uD_up_value = np.vstack([u_up,v_up])
        uD_up_fun = MyExpression_top(X[coor_up], Y[coor_up], uD_up_value, mesh2.geometry.dim)
        uD_up.interpolate(uD_up_fun.eval)
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, up_line)
        bc_up = fem.dirichletbc(uD_up, boundary_dofs)

        # Set up boundary condition at low
        uD_low = Function(V2)
        uD_low_value = np.vstack([u_low,v_low])
        uD_low_fun = MyExpression_low(X[coor_low], Y[coor_low], uD_low_value, mesh2.geometry.dim)
        uD_low.interpolate(uD_low_fun.eval)
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, low_line)
        bc_low = fem.dirichletbc(uD_low, boundary_dofs)

        # Set up boundary condition at left
        uD_left = Function(V2)
        uD_left_value = np.vstack([u_left,v_left])
        uD_left_fun = MyExpression_left(X[coor_left], Y[coor_left], uD_left_value, mesh2.geometry.dim)
        uD_left.interpolate(uD_left_fun.eval)
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, left_line)
        bc_left = fem.dirichletbc(uD_left, boundary_dofs)

        # Set up boundary condition at right
        uD_right = Function(V2)
        uD_right_value = np.vstack([u_right,v_right])
        uD_right_fun = MyExpression_right(X[coor_right], Y[coor_right], uD_right_value, mesh2.geometry.dim)
        uD_right.interpolate(uD_right_fun.eval)
        boundary_dofs = fem.locate_dofs_topological(V2, fdim2, right_line)
        bc_right = fem.dirichletbc(uD_right, boundary_dofs)

        bc2 = [bc_up, bc_low, bc_left, bc_right]

        if (ts == 0 and iter >=1) or (ts >= 1 and ts < ts_end):
            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            '''
            =========================================================
            Atthenion: before convergence, this part is used to 
            assign the stored values back to u_old, v_old, and a_old
            instead of stepping forward 
            =========================================================
            '''
            u_old2.x.petsc_vec.setArray(u_old_1)
            v_old2.x.petsc_vec.setArray(v_old_1)
            a_old2.x.petsc_vec.setArray(a_old_1)

        if ts >= ts_end:
            # Assign the stored values back to u_old, v_old, and a_old (- delta_t)
            # ts_end is the FE-FE coupling end time step
            # in FE-NO, u,v and a are no longer petsc vectors
            u_old2 = np.copy(u_old_1)
            v_old2 = np.copy(v_old_1)
            a_old2 = np.copy(a_old_1)



        if ts < ts_end:   
            #region Second FE solver      
            LL2 = rho*ufl.inner(2*(du2) / (beta * dt**2), u_2)*dx + \
                ufl.inner(sigma(du2), ufl.sym(ufl.grad(u_2)))*dx    
                
            RR2 = rho*ufl.inner(2*( u_old2 + dt * v_old2) / (beta * dt**2) + \
                                (1 -  beta) / beta* a_old2 , u_2)*dx 
                
            a_form2 = fem.form(LL2)
            L_form2 = fem.form(RR2)

            A2 = assemble_matrix(a_form2, bcs=bc2)
            A2.assemble()
            b2 = create_vector(L_form2)
            solver = PETSc.KSP().create(mesh2.comm)
            solver.setOperators(A2)
            solver.setType(PETSc.KSP.Type.CG)
            solver.getPC().setType(PETSc.PC.Type.JACOBI)
            
            with b2.localForm() as loc_b2:
                loc_b2.set(0)
            assemble_vector(b2, L_form2)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b2, [a_form2], [bc2])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, bc2)
            # Solve linear problem
            solver.solve(b2, u2.x.petsc_vec)
            u2.x.scatter_forward()
            
            # Store the current values of u_old, v_old, a_old
            u_old_1 = np.copy(u_old2.x.petsc_vec.array)
            v_old_1 = np.copy(v_old2.x.petsc_vec.array)
            a_old_1 = np.copy(a_old2.x.petsc_vec.array)
            
            # Update old fields with new quantities
            # Attention: we use stored values u_old_1, v_old_1, a_old_1 in one inner iteration
            update_fields(u2, u_old2, v_old2, a_old2)
                
            topology, u_cell_types, u_geometry = plot.vtk_mesh(V2)
            u_values = u2.x.array.real
            u_tot = u_values.reshape(-1,3)
            U_1, V_1= u_tot[:,0], u_tot[:,1]
            X1, Y1 = u_geometry[:,0], u_geometry[:,1]

            # retrieve stress
            gdim = mesh2.geometry.dim
            ### stress is 3*3  (gdim, gdim)
            Function_space_for_sigma = fem.functionspace(mesh2, ("CG", 1, (gdim, gdim))) 
            expr= fem.Expression(sigma(u2), Function_space_for_sigma.element.interpolation_points())        
            sigma_values = Function(Function_space_for_sigma) 
            sigma_values.interpolate(expr)
            X1 = mesh2.geometry.x[:,0]
            Y1 = mesh2.geometry.x[:,1]
            sigma_tot1 = sigma_values.x.array.reshape(-1,9)
            
            
            # displacement on the four boundaries (inner domain inner edges)
            u_up_in = U_1[coor_up_in].reshape(1,-1)
            v_up_in = V_1[coor_up_in].reshape(1,-1)

            u_low_in = U_1[coor_low_in].reshape(1,-1)
            v_low_in = V_1[coor_low_in].reshape(1,-1)

            u_left_in = U_1[coor_left_in].reshape(1,-1)
            v_left_in = V_1[coor_left_in].reshape(1,-1)

            u_right_in = U_1[coor_right_in].reshape(1,-1)
            v_right_in = V_1[coor_right_in].reshape(1,-1)

            uD_up_value_in = np.vstack([u_up_in,v_up_in])
            uD_low_value_in = np.vstack([u_low_in,v_low_in])
            uD_left_value_in = np.vstack([u_left_in,v_left_in])
            uD_right_value_in = np.vstack([u_right_in,v_right_in]) 

            # locate the outer domain inner edges in FEniCSX
            up_line_in = dolfinx.mesh.locate_entities_boundary(mesh, fdim2, top_inner_in)
            low_line_in = dolfinx.mesh.locate_entities_boundary(mesh, fdim2, low_inner_in)
            left_line_in = dolfinx.mesh.locate_entities_boundary(mesh, fdim2, left_inner_in)
            right_line_in = dolfinx.mesh.locate_entities_boundary(mesh, fdim2, right_inner_in)
           
            # Use Rbf interpolation to get the displacement values on the outer domain inner edges
            # Use interpolation to avoid the mismatch of points on deges from two subdomains
            u_up_fun = Rbf(x_up_in0, y_up_in0, u_up_in0)
            v_up_fun = Rbf(x_up_in0, y_up_in0, v_up_in0)
            u_up_value = u_up_fun(x_up_in, y_up_in)
            v_up_value = v_up_fun(x_up_in, y_up_in)
            uD_up_new = np.concatenate([u_up_value.reshape(1,-1) , v_up_value.reshape(1,-1)], axis = 0)

            # Relaxation formula to get new displacment values 
            uD_up_tot = theta * uD_up_value_in + (1-theta)*uD_up_new

            # Create a FEniCSX expression for the new displacement values
            # and interpolate it to the function space V to set bounary conditions for outer domain
            uD_up_out_fun = MyExpression_top_in(x_up_in, y_up_in, uD_up_tot, mesh.geometry.dim)
            uD_up_out = Function(V)
            uD_up_out.interpolate(uD_up_out_fun.eval)
            boundary_dofs_out = fem.locate_dofs_topological(V, fdim2, up_line_in)
            bc_up_out = fem.dirichletbc(uD_up_out, boundary_dofs_out)


            u_low_fun = Rbf(x_low_in0, y_low_in0, u_low_in0)
            v_low_fun = Rbf(x_low_in0, y_low_in0, v_low_in0)
            u_low_value = u_low_fun(x_low_in, y_low_in)
            v_low_value = v_low_fun(x_low_in, y_low_in)
            uD_low_new = np.concatenate([u_low_value.reshape(1,-1) , v_low_value.reshape(1,-1)], axis = 0)

            uD_low_tot = theta * uD_low_value_in + (1-theta)*uD_low_new

            uD_low_out_fun = MyExpression_low_in(x_low_in, y_low_in, uD_low_tot, mesh.geometry.dim)
            uD_low_out = Function(V)
            uD_low_out.interpolate(uD_low_out_fun.eval)
            boundary_dofs_out = fem.locate_dofs_topological(V, fdim2, low_line_in)
            bc_low_out = fem.dirichletbc(uD_low_out, boundary_dofs_out)


            u_left_fun = Rbf(x_left_in0, y_left_in0, u_left_in0)
            v_left_fun = Rbf(x_left_in0, y_left_in0, v_left_in0)
            u_left_value = u_left_fun(x_left_in, y_left_in)
            v_left_value = v_left_fun(x_left_in, y_left_in)
            uD_left_new = np.concatenate([u_left_value.reshape(1,-1) , v_left_value.reshape(1,-1)], axis = 0)

            uD_left_tot = theta * uD_left_value_in + (1-theta)*uD_left_new

            uD_left_out_fun = MyExpression_left_in(x_left_in, y_left_in, uD_left_tot, mesh.geometry.dim)
            uD_left_out = Function(V)
            uD_left_out.interpolate(uD_left_out_fun.eval)
            boundary_dofs_out = fem.locate_dofs_topological(V, fdim2, left_line_in)
            bc_left_out = fem.dirichletbc(uD_left_out, boundary_dofs_out)


            u_right_fun = Rbf(x_right_in0, y_right_in0, u_right_in0)
            v_right_fun = Rbf(x_right_in0, y_right_in0, v_right_in0)
            u_right_value = u_right_fun(x_right_in, y_right_in)
            v_right_value = v_right_fun(x_right_in, y_right_in)
            uD_right_new = np.concatenate([u_right_value.reshape(1,-1) , v_right_value.reshape(1,-1)], axis = 0)

            uD_right_tot = theta * uD_right_value_in + (1-theta)*uD_right_new

            uD_right_out_fun = MyExpression_right_in(x_right_in, y_right_in, uD_right_tot, mesh.geometry.dim)
            uD_right_out = Function(V)
            uD_right_out.interpolate(uD_right_out_fun.eval)
            boundary_dofs_out = fem.locate_dofs_topological(V, fdim2, right_line_in)
            bc_right_out = fem.dirichletbc(uD_right_out, boundary_dofs_out)

            # new bounary conditions for all the eges in the outer domain (Key point for FE-FE coupling)
            bc = [bc_bt, bc_l, bc_top, bc_r, bc_up_out, bc_low_out, bc_left_out, bc_right_out]

    
        if ts >=ts_end: 
            # region Inner NN 
            '''
            =============================================================
            Each parmas serve for 10 time steps (can be further extended)
            =============================================================
            '''
            if ts >= ts_end+10:
                params = params0

            if ts >= ts_end+20:
                params = params1

            if ts >= ts_end+30:
                params = params2

            if ts >= ts_end+40:
                params = params3

            iter_NN += 1
            # predict the displacement for the whole domain
            y_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])

            # get the displacement values on the outer domain outer edges
            u_l_out, v_l_out = U_[index_left_out], V_[index_left_out]
            u_r_out, v_r_out = U_[index_right_out], V_[index_right_out]
            u_up_out, v_up_out = U_[index_up_out], V_[index_up_out]
            u_down_out, v_down_out = U_[index_down_out], V_[index_down_out]

            # Rbf interpolation to get the displacement values on the outer domain outer edges (avoid mismatch)
            u_l_real = Rbf(X[index_left_out], Y[index_left_out], u_l_out)(X1[index_left], Y1[index_left]).reshape(1,-1)
            v_l_real = Rbf(X[index_left_out], Y[index_left_out], v_l_out)(X1[index_left], Y1[index_left]).reshape(1,-1)
            u_r_real = Rbf(X[index_right_out], Y[index_right_out], u_r_out)(X1[index_right], Y1[index_right]).reshape(1,-1)
            v_r_real = Rbf(X[index_right_out], Y[index_right_out], v_r_out)(X1[index_right], Y1[index_right]).reshape(1,-1)
            u_up_real = Rbf(X[index_up_out], Y[index_up_out], u_up_out)(X1[index_up], Y1[index_up]).reshape(1,-1)
            v_up_real = Rbf(X[index_up_out], Y[index_up_out], v_up_out)(X1[index_up], Y1[index_up]).reshape(1,-1)
            u_low_real = Rbf(X[index_down_out], Y[index_down_out], u_down_out)(X1[index_down], Y1[index_down]).reshape(1,-1)
            v_low_real = Rbf(X[index_down_out], Y[index_down_out], v_down_out)(X1[index_down], Y1[index_down]).reshape(1,-1)

            
            u_old_, v_old_, vx_old, vy_old = u_old2.reshape(-1,3)[:, 0], u_old2.reshape(-1,3)[:, 1],\
                                                v_old2.reshape(-1,3)[:, 0], v_old2.reshape(-1,3)[:, 1]

            # interpolation to get input of Branch1 (domain values of displacement and velocity, as input of CNN structured grid)
            u_old_ = Rbf(X1, Y1, u_old_)(X1_, Y1_)
            v_old_ = Rbf(X1, Y1, v_old_)(X1_, Y1_)
            vx_old = Rbf(X1, Y1, vx_old)(X1_, Y1_)
            vy_old = Rbf(X1, Y1, vy_old)(X1_, Y1_)
            
            # input of Branch2 (boundary conditions)
            u_bc = np.hstack([u_l_real, u_r_real, u_up_real, u_low_real])
            v_bc = np.hstack([v_l_real, v_r_real, v_up_real, v_low_real])

            # the scale factor 100 is used to match the scale of the input data    
            u_test = np.hstack([u_old_.reshape(1,-1), v_old_.reshape(1,-1), vx_old.reshape(1,-1)/100, vy_old.reshape(1,-1)/100, u_bc.reshape(1,-1), v_bc.reshape(1,-1)]) 
            v_test = np.hstack([u_old_.reshape(1,-1), v_old_.reshape(1,-1), vx_old.reshape(1,-1)/100, vy_old.reshape(1,-1)/100, u_bc.reshape(1,-1), v_bc.reshape(1,-1)]) 
            s_u_pred, s_v_pred = model.predict_s(params, u_test, v_test, y_test)
            sigma_sx_pred, sigma_sy_pred, sigma_sxy_pred = model.stress(params, u_test, v_test, y_test)

            # Store the current values of u_old, v_old, a_old
            u_old_1 = np.copy(u_old2)
            v_old_1 = np.copy(v_old2)
            a_old_1 = np.copy(a_old2)
            
            u2_NN = np.hstack((s_u_pred.reshape(-1,1), s_v_pred.reshape(-1,1), np.zeros(s_u_pred.shape).reshape(-1,1))).flatten()
            # Update old fields with new quantities obtained from NN 
            # Attention: we use stored values u_old_1, v_old_1, a_old_1 before convergence 
            
            update_fields_NN(u2_NN, u_old2, v_old2, a_old2)

            # the bounray coordinates for the inner domain inner edges
            y_test_up_in = np.hstack([x_up_in.reshape(-1,1), y_up_in.reshape(-1,1)])
            y_test_low_in = np.hstack([x_low_in.reshape(-1,1), y_low_in.reshape(-1,1)])
            y_test_left_in = np.hstack([x_left_in.reshape(-1,1), y_left_in.reshape(-1,1)])
            y_test_right_in = np.hstack([x_right_in.reshape(-1,1), y_right_in.reshape(-1,1)])

            # predict the displacement for the inner domain inner edges by DeepONet 
            s_u_pred_up_in, s_v_pred_up_in = model.predict_s(params, u_test, v_test, y_test_up_in)
            s_u_pred_low_in, s_v_pred_low_in = model.predict_s(params, u_test, v_test, y_test_low_in)
            s_u_pred_left_in, s_v_pred_left_in = model.predict_s(params, u_test, v_test, y_test_left_in)
            s_u_pred_right_in, s_v_pred_right_in = model.predict_s(params, u_test, v_test, y_test_right_in)

            # Use Rbf interpolation to get the displacement values on the outer domain inner edges
            # Use interpolation to avoid the mismatch of points on deges from two subdomains
            uD_up_inner_value = np.vstack([s_u_pred_up_in.reshape(1,-1), s_v_pred_up_in.reshape(1,-1)])
            u_fun_up_in = Rbf(x_up_in0, y_up_in0, u_up_in0)
            v_fun_up_in = Rbf(x_up_in0, y_up_in0, v_up_in0)
            u_out_up_in = u_fun_up_in(x_up_in, y_up_in)
            v_out_up_in = v_fun_up_in(x_up_in, y_up_in)    
            uD_up_new = np.concatenate([u_out_up_in.reshape(1,-1) , v_out_up_in.reshape(1,-1)], axis = 0)

            # Relaxation formula to get new displacment values
            uD_up_tot = theta * uD_up_inner_value + (1-theta)*uD_up_new

            # Create a FEniCSX expression for the new displacement values
            # and interpolate it to the function space V to set bounary conditions for outer domain 
            uD_up_out_fun = MyExpression_top_in(x_up_in, y_up_in, uD_up_tot, mesh.geometry.dim)
            uD_up_out.interpolate(uD_up_out_fun.eval)
            boundary_dofs = fem.locate_dofs_topological(V, fdim, up_line_in)
            bc_up_out = fem.dirichletbc(uD_up_out, boundary_dofs)

            # for bottom boundary 
            uD_low_inner_value = np.vstack([s_u_pred_low_in.reshape(1,-1), s_v_pred_low_in.reshape(1,-1)])
            u_fun_low_in = Rbf(x_low_in0, y_low_in0, u_low_in0)
            v_fun_low_in = Rbf(x_low_in0, y_low_in0, v_low_in0)
            u_out_low_in = u_fun_low_in(x_low_in, y_low_in)
            v_out_low_in = v_fun_low_in(x_low_in, y_low_in)
            uD_low_new = np.concatenate([u_out_low_in.reshape(1,-1) , v_out_low_in.reshape(1,-1)], axis = 0)

            uD_low_tot = theta * uD_low_inner_value + (1-theta)*uD_low_new

            uD_low_out_fun = MyExpression_low_in(x_low_in, y_low_in, uD_low_tot, mesh.geometry.dim)
            uD_low_out.interpolate(uD_low_out_fun.eval)
            boundary_dofs = fem.locate_dofs_topological(V, fdim, low_line_in)
            bc_low_out = fem.dirichletbc(uD_low_out, boundary_dofs)

            # for left boundary
            uD_left_inner_value = np.vstack([s_u_pred_left_in.reshape(1,-1), s_v_pred_left_in.reshape(1,-1)])
            u_fun_left_in = Rbf(x_left_in0, y_left_in0, u_left_in0)
            v_fun_left_in = Rbf(x_left_in0, y_left_in0, v_left_in0)
            u_out_left_in = u_fun_left_in(x_left_in, y_left_in)
            v_out_left_in = v_fun_left_in(x_left_in, y_left_in)
            uD_left_new = np.concatenate([u_out_left_in.reshape(1,-1) , v_out_left_in.reshape(1,-1)], axis = 0)

            uD_left_tot = theta * uD_left_inner_value + (1-theta)*uD_left_new

            uD_left_out_fun = MyExpression_left_in(x_left_in, y_left_in, uD_left_tot, mesh.geometry.dim)
            uD_left_out.interpolate(uD_left_out_fun.eval)
            boundary_dofs = fem.locate_dofs_topological(V, fdim, left_line_in)
            bc_left_out = fem.dirichletbc(uD_left_out, boundary_dofs)

            # for right boundary
            uD_right_inner_value = np.vstack([s_u_pred_right_in.reshape(1,-1), s_v_pred_right_in.reshape(1,-1)])
            u_fun_right_in = Rbf(x_right_in0, y_right_in0, u_right_in0)
            v_fun_right_in = Rbf(x_right_in0, y_right_in0, v_right_in0)
            u_out_right_in = u_fun_right_in(x_right_in, y_right_in)
            v_out_right_in = v_fun_right_in(x_right_in, y_right_in)
            uD_right_new = np.concatenate([u_out_right_in.reshape(1,-1) , v_out_right_in.reshape(1,-1)], axis = 0)

            uD_right_tot = theta * uD_right_inner_value + (1-theta)*uD_right_new

            uD_right_out_fun = MyExpression_right_in(x_right_in, y_right_in, uD_right_tot, mesh.geometry.dim)
            uD_right_out.interpolate(uD_right_out_fun.eval)
            boundary_dofs = fem.locate_dofs_topological(V, fdim, right_line_in)
            bc_right_out = fem.dirichletbc(uD_right_out, boundary_dofs)

            # new bounary conditions for all the eges in the outer domain (Key point for FE-NO coupling)
            bc = [bc_bt, bc_l, bc_top, bc_r, bc_up_out, bc_low_out, bc_left_out, bc_right_out]
  

        U_copy = np.copy(U_)
        V_copy = np.copy(V_)
        U_1_copy = np.copy(U_1)
        V_1_copy = np.copy(V_1)

        u_list.append(U_copy)
        v_list.append(V_copy)
        u2_list.append(U_1_copy)
        v2_list.append(V_1_copy)

        break_list.append(sigma_tot[:,0])
        if len(u_list) > 1:
            # L2 error for U and V in outer and inner domains 
            uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
            uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
            print('\n' ,'error', uv_L2 + uv_L2_2)
            error_list.append(uv_L2 + uv_L2_2)

            if  error_list[-1] < 1e-5:
                    
                if ts >= ts_end and (ts - ts_end) % 5 == 0:
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,0],'U_x_FE t=' + str(ts) + ' iter = ' + str(iter), rf'$u_{{x, \mathrm{{FE-NO}},\Omega_{{I}}}}^{{{ts},{iter}}}$')
                    plot_disp(X, Y, u.x.array.reshape(-1,3)[:,1],'U_y_FE t=' + str(ts) + ' iter = ' + str(iter), rf'$u_{{y, \mathrm{{FE-NO}},\Omega_{{I}}}}^{{{ts},{iter}}}$')
                    plot_disp_real(X1, Y1, s_u_pred,'U_x_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$u_{{x,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    plot_disp_real(X1, Y1, s_v_pred,'U_y_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$u_{{y,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    plot_disp(X1, Y1, sigma_sx_pred,'stress_x_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$\sigma_{{xx,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    plot_disp(X1, Y1, sigma_sy_pred,'stress_y_NN t=' + str(ts) + ' iter = ' + str(iter), rf'$\sigma_{{yy,\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{ts},{iter}}}$')
                    
                    # Save the results to files
                    '''np.savetxt('U ts = ' + str(ts) +'.txt', U_)
                    np.savetxt('V ts = ' + str(ts) +'.txt', V_)
                    np.savetxt('V1 ts = ' + str(ts) +'.txt', s_v_pred)  
                    np.savetxt('U1 ts = ' + str(ts) +'.txt', s_u_pred)
                    np.savetxt('sigma_sx_pred ts = ' + str(ts) +'.txt', sigma_sx_pred)
                    np.savetxt('sigma_sy_pred ts = ' + str(ts) +'.txt', sigma_sy_pred)
                    np.savetxt(f'error_list_FE_NO_elasto ts=' + str(ts) + ' iter=' + str(iter) + '.txt', error_list)'''
                
                # Attention: only meet the tolerance, we can break the inner iteration
                # and the stored values are updated to the new values (+ delta_t)
                u_old_0 = np.copy(u_old.x.petsc_vec.array)
                v_old_0 = np.copy(v_old.x.petsc_vec.array)
                a_old_0 = np.copy(a_old.x.petsc_vec.array)

                if ts < ts_end:
                    # use in fenicsx
                    u_old_1 = np.copy(u_old2.x.petsc_vec.array)
                    v_old_1 = np.copy(v_old2.x.petsc_vec.array)
                    a_old_1 = np.copy(a_old2.x.petsc_vec.array)                    
                       
                else:
                    # use in NN
                    u_old_1 = np.copy(u_old2)
                    v_old_1 = np.copy(v_old2)
                    a_old_1 = np.copy(a_old2)

                break

