'''
======================================================================
The FE-DeepONet or FE-NO coupling framework for linear elastic materials 
under static loading conditions.
----------------------------------------------------------------------
The overlapping boundary is used. 
The domain decomposition and Schwartz alternating method is used by
exchanging the displacement at the overlapping boundary.
======================================================================
'''

import os
import jax
import jax.numpy as jnp
from jax import grad, vmap
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh
from dolfinx.fem import functionspace, Function 
from dolfinx import mesh
from dolfinx import fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import pyvista
import matplotlib.pyplot as plt
from dolfinx import plot
from dolfinx import io
from pathlib import Path
import pickle
from scipy.interpolate import Rbf, interp1d, griddata
import math 
from matplotlib.ticker import ScalarFormatter
import time
from utils import plot_disp, plot_relative_error, createFolder


#region Save path       
originalDir = os.getcwd()
print('current working directory is ' + originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_DeepONet_static_coupling_results'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

originalDir_real = os.path.join(originalDir, './'+ foldername + '/')
#### Gmsh generates the geometry 
lc = 0.02

# region gmsh
#===================================================
# Using Gmsh to generate two regions: 
# 1. Outer region with a square and a hole (FE region)
# 2. Inner region with a disk (DeepONet region)
# overlapping region exists between the two regions
#---------------------------------------------------
# Inner region mesh is not used for FE, it provides
# the coordinates for the DeepONet model
#===================================================

# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
radius1 = 0.3 - 0.05
num_points = 200      # Number of points on the circumference #so important !!!0 
num_points1 = 200  

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

points1 = []
for i in range(num_points):
    angle1 = 2 * math.pi * i / num_points
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)
    points1.append(gmsh.model.occ.addPoint(x1, y1, center[2], lc))

# Create lines between consecutive points
lines1 = []
for i in range(num_points):
    p1 = points1[i]
    p2 = points1[(i + 1) % num_points]
    lines1.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop1 = gmsh.model.occ.addCurveLoop(lines1)

# Create a surface from the loop
surface_hole1 = gmsh.model.occ.addPlaneSurface([circle_loop1])

gmsh.model.occ.synchronize()

# cut the hole 
# Define the points for the circle
points = []
for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = center[0] + radius1 * math.cos(angle)
    y = center[1] + radius1 * math.sin(angle)
    points.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines = []
for i in range(num_points):
    p1 = points[i]
    p2 = points[(i + 1) % num_points]
    lines.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop = gmsh.model.occ.addCurveLoop(lines)

# Create a surface from the loop
surface_hole = gmsh.model.occ.addPlaneSurface([circle_loop])

# Cut the larger square with the hole
gmsh.model.occ.cut([(2, background)], [(2, surface_hole)])
# Synchronize the model to GMSH
gmsh.model.occ.synchronize()

# Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2, background)],  # Target entities
    [(2, surface_hole1)]  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [background], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surface_hole1], tag=2)
# Generate 2D mesh
gmsh.model.mesh.generate(2)

gmsh.model.addPhysicalGroup(1, [line1, line2, line3, line4], name="LargerSquareEdges") # facet tag = 7
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges") # facet tag = 8 interupt the boundary 
gmsh.model.addPhysicalGroup(1, lines, name="smallHoleEdges")# facet tag = 9



# Write the mesh to a file (optional)
gmsh.write("outer_region.msh")

# Finalize GMSH
gmsh.finalize()



# region Hole 
# initialize GMSH
gmsh.initialize()
gmsh.model.add("model")
# Define inner square
# add the square points
points1 = []
for i in range(num_points):
    angle1 = 2 * math.pi * i / num_points
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)
    points1.append(gmsh.model.occ.addPoint(x1, y1, center[2], lc))

# Create lines between consecutive points
lines1 = []
for i in range(num_points):
    p1 = points1[i]
    p2 = points1[(i + 1) % num_points]
    lines1.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop1 = gmsh.model.occ.addCurveLoop(lines1)

# Create a surface from the loop
surface_hole1 = gmsh.model.occ.addPlaneSurface([circle_loop1])


gmsh.model.occ.synchronize()

# cut the hole 
# Define the points for the circle
points = []
for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = center[0] + radius1 * math.cos(angle)
    y = center[1] + radius1 * math.sin(angle)
    points.append(gmsh.model.occ.addPoint(x, y, center[2], lc))

# Create lines between consecutive points
lines = []
for i in range(num_points):
    p1 = points[i]
    p2 = points[(i + 1) % num_points]
    lines.append(gmsh.model.occ.addLine(p1, p2))

# Create a closed loop
circle_loop = gmsh.model.occ.addCurveLoop(lines)

# Create a surface from the loop
surface_hole = gmsh.model.occ.addPlaneSurface([circle_loop])

    # Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2, surface_hole1)],  # Target entities
    [(2, surface_hole)]  # Tool entities
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
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, lines, name="HoleEdges") # facet tag = 4
gmsh.write("inner_hole.msh")

# Finalize GMSH
gmsh.finalize()

import meshio

# read .msh file
msh = meshio.read("outer_region.msh")
# extract points and cell information
points = msh.points
cells = msh.cells_dict["triangle"]  
plt.figure(figsize=(8, 8))

#plot every triangle 
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5)  # 用黑色线条绘制

#set axis 
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization2')
plt.savefig('outer_region' + ".jpg", dpi=700)
plt.show()

# read .msh file 
msh = meshio.read("inner_hole.msh")
# extract points and cell information
points = msh.points
cells = msh.cells_dict["triangle"]  
# plot the mesh 
plt.figure(figsize=(8, 8))

# plot every triangle
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-',linewidth=0.5)  # 用黑色线条绘制

plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization')
plt.savefig('inner_hole' + ".jpg", dpi=700)
plt.show()

#region FEM 
import dolfinx 
# import gmsh file
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("outer_region.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 1, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 

print('mesh number', mesh1.topology.index_map(2).size_local)
print('DOF number', V.dofmap.index_map.size_local)

### linear materials parameters 
E = 0.210e-2 # Young's modulus 
nu = 0.3    # Poisson's ratio

# strain function in ufl(a domain specific language for declaration of finite element discretizations of variational forms)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# stress function in ufl
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))

# Find the boudnary and define the boundary conditions for outer region 
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1
upper_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.5))
uD = np.array([0, 0.01, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_point)
bc_upper = fem.dirichletbc(uD, boundary_dofs, V)

upper_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_point)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

bcs = [bc_upper, bc_bt]


def on_cricle(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    return xc, yc 

def on_cricle_inner(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))]
    return xc, yc 


# region Mypression
#==================================================
# MyExpression is used to interpolate the displacement at the boundary
#==================================================

class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        
        return values 

class MyExpression_inner_hole:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        
        values = np.zeros((self.V_dim, x.shape[1]))
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 


    
# define the variational problem in FEM 
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

# outer hole boundary
u_geometry = mesh1.geometry.x
X, Y = u_geometry[:,0], u_geometry[:,1]
x_c_out, y_c_out = on_cricle(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_c_in, y_c_in = on_cricle_inner(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

coor_r_out = np.array([np.where((X == x_val) & (Y == y_val))[0] 
                    for x_val, y_val in zip(x_c_out, y_c_out)])[:,0]

coor_r_in = np.array([np.where((X == x_val) & (Y == y_val))[0]
                    for x_val, y_val in zip(x_c_in, y_c_in)])[:,0]

# inner hole boundary
mesh2, cell_markers, facet_markers  = gmshio.read_from_msh("inner_hole.msh", MPI.COMM_WORLD) 
V2 = functionspace(mesh2, ("CG", 1, (mesh2.geometry.dim, ))) 
u_topology1, u_cell_types1, u_geometry1 = plot.vtk_mesh(V2)
X1, Y1 = u_geometry1[:,0], u_geometry1[:,1]
x_c1_out, y_c1_out = on_cricle(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))
x_c1_in, y_c1_in = on_cricle_inner(np.vstack([X1.reshape(1,-1), Y1.reshape(1,-1)]))

coor_r1_out = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0] 
                    for x_val, y_val in zip(x_c1_out, y_c1_out)])[:,0]

coor_r1_in = np.array([np.where((X1 == x_val) & (Y1 == y_val))[0]
                    for x_val, y_val in zip(x_c1_in, y_c1_in)])[:,0]

np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)
np.savetxt('xc1_out.txt', x_c1_out)
np.savetxt('yc1_out.txt', y_c1_out)
np.savetxt('xc1_in.txt', x_c1_in)
np.savetxt('yc1_in.txt', y_c1_in)

# region DeepONet 
#=========================================================
# Pretrained DeepONet model is used to predict the displacement
# for the inner region
#=========================================================


from Prepare_DeepONet_static import PI_DeepONet
m = num_points1 
d = 2 # dimension of the input data
ela_model = dict()
ela_model['E'] = E #1000 
ela_model['nu'] = nu

branch_layers =  [2*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)

os.chdir(os.path.join(originalDir, './' + 'Pretrained_DeepONet_static' + '/'))
print(os.getcwd())

# load the pretrained model parameters
with open('DeepONet_static.pkl', 'rb') as f:
    params = pickle.load(f)

os.chdir(originalDir_real)


# resort the sensor points on the boundary
num_points = m           # m is the sensor number  
x_list = []
y_list = []
for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    x_list.append(x)
    y_list.append(y)  

x = np.array(x_list).reshape(1,-1)
y = np.array(y_list).reshape(1,-1)
index = []
# for resort data to use in the sensor points 

for i in range(x.shape[1]):
    index_1 = np.where(np.isclose(x_c1_out.reshape(1,-1),x[0,i])&np.isclose(y_c1_out.reshape(1,-1),y[0,i]))[1]
    index.append(index_1)




start_time = time.time()

u_list, v_list, u2_list, v2_list =[],[],[],[]
break_list = []
error_list = []
#region Coupling 
niter = 1000
theta = 0.5 # relaxation coefficient
#### Coupling  
for iter in range(0, niter):
    ## FEM 
    if iter == 0:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()### first iteration, assume T at interface = 0 
        print('success initiation')
    
    else:     
        #ds = ufl.Measure("ds", domain=mesh1) 
        f = fem.Constant(mesh1, default_scalar_type((0, 0, 0)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma(u), epsilon(v))*ufl.dx
        L = ufl.dot(f, v)*ufl.dx 

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
                
        print('success loop =' + f'{iter}')
        
    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    u, v, w= u_tot[:,0], u_tot[:,1], u_tot[:,2]
    
    # Extract displacment at outer circle 
    u_c = u[coor_r_out].reshape(1,-1)
    v_c = v[coor_r_out].reshape(1,-1)
    
    gdim = mesh1.geometry.dim

    # region handshake
    # interpolation the displacement from the outer region to the inner region
    uc_fun = Rbf(x_c_out, y_c_out, u_c)
    vc_fun = Rbf(x_c_out, y_c_out, v_c)
    u2c = uc_fun(x_c1_out, y_c1_out)[index].reshape(1,-1)
    v2c = vc_fun(x_c1_out, y_c1_out)[index].reshape(1,-1)

    u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
    v_test = np.hstack([u2c, v2c]).reshape(1,-1)
    hc_test = np.hstack([X1[coor_r1_in].reshape(-1,1), Y1[coor_r1_in].reshape(-1,1)])
    # predict the displacement at the inner boundary
    #=================================================
    # the u_test and v_test are the displacement at
    # the outer boundary as boundary condition
    # the hc_test is the coordinate of the inner boundary
    #=================================================
    s_uc_pred_in, s_vc_pred_in = model.predict_s(params, u_test, v_test, hc_test)

    #displacement at inner boundary 
    u_c2 = s_uc_pred_in.reshape(1,-1)
    v_c2 = s_vc_pred_in.reshape(1,-1)
    gdim = mesh2.geometry.dim

    ### stress is 3*3  (gdim, gdim)
    '''Function_space_for_sigma = fem.functionspace(mesh2, ("CG", 1, (gdim, gdim)))        
    expr= fem.Expression(sigma(uh2), Function_space_for_sigma.element.interpolation_points())        
    sigma_values = Function(Function_space_for_sigma) 
    sigma_values.interpolate(expr)
    X1 = mesh2.geometry.x[:,0]
    Y1 = mesh2.geometry.x[:,1]
    sigma_tot1 = sigma_values.x.array.reshape(-1,9)'''

    #==================================================
    # Define the inner circle displacement as the boundary
    # condition for the outer region
    #==================================================

    fdim2 = mesh2.topology.dim -1 
    c_line_in = dolfinx.mesh.locate_entities_boundary(mesh1, fdim2, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))
    uD_c2 = Function(V)
    uD_c2_value = np.vstack([u_c2,v_c2])
    # use MyExpression_inner_hole to interpolate the displacement at the inner boundary (from inner hole to outer region)
    uD_c2_fun = MyExpression_inner_hole(X1[coor_r1_in], Y1[coor_r1_in], uD_c2_value, mesh1.geometry.dim)
    uD_c2.interpolate(uD_c2_fun.eval)        
    boundary_dofs2 = fem.locate_dofs_topological(V, fdim2, c_line_in)
    bc_c2 = fem.dirichletbc(uD_c2, boundary_dofs2)            
    bcs  = [bc_upper, bc_bt, bc_c2] # update the bounary conditions with the inner boundary condition 
    h_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
    s_u_pred_tot, s_v_pred_tot = model.predict_s(params, u_test, v_test, h_test)

    
    if iter ==0 :
        np.savetxt('X.txt', X)
        np.savetxt('Y.txt', Y)
        np.savetxt('X1.txt', X1)
        np.savetxt('Y1.txt', Y1)      

    if iter == 0 or iter == 5 or iter == 11:
        plot_disp(X1,Y1,s_u_pred_tot,'u inner i = ' + str(iter),  rf'$u_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X1,Y1,s_v_pred_tot,'v inner i = ' + str(iter),  rf'$v_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{iter}}}$')
        plot_disp(X,Y,u, 'u outer i = ' + str(iter), rf'$u_{{\mathrm{{FE-NO, \Omega_I}}}}^{{{iter}}}$')
        plot_disp(X,Y,v, 'v outer i = ' + str(iter), rf'$v_{{\mathrm{{FE-NO, \Omega_I}}}}^{{{iter}}}$')

        np.savetxt('u2 i = '+ str(iter) + ' .txt', s_u_pred_tot)
        np.savetxt('v2 i = '+ str(iter) + ' .txt', s_v_pred_tot)
        np.savetxt('u i = '+ str(iter) + ' .txt', u)
        np.savetxt('v i = '+ str(iter) + ' .txt', v)

    u_list.append(u)
    v_list.append(v)
    u2_list.append(s_u_pred_tot)
    v2_list.append(s_v_pred_tot)

    if len(u_list) > 1:
        uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
        uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
        print('\n' ,'error', uv_L2 + uv_L2_2)
        error_list.append(uv_L2 + uv_L2_2)
        if error_list[-1] < 1e-3:
            # L2 error is smaller than 1e-3, break the loop
            break

np.save('error_list_FE_NN', error_list)    
end_time = time.time()  
print('time', end_time - start_time)

# region plot error
#=========================================================
# Plot the error between the predicted displacement and the ground truth
# Ground truth data is generated by FEM in full square
#=========================================================

try:
    os.chdir(os.path.join(originalDir, './'+ 'static_data_ground_truth' + '/'))
    # load the ground truth data generated by FEM in full sqaure 
    U_full_FE = np.loadtxt('u.txt')
    V_full_FE = np.loadtxt('v.txt')
    X_FE, Y_FE = np.loadtxt('X.txt'), np.loadtxt('Y.txt')
    # interpolate the ground truth data to the sensor points
    U_func = Rbf(X_FE, Y_FE, U_full_FE)
    V_func = Rbf(X_FE, Y_FE, V_full_FE)

    # inner region data
    U1_FE = U_func(X1, Y1)
    V1_FE = V_func(X1, Y1)
    # outer region data
    U_FE = U_func(X, Y)
    V_FE = V_func(X, Y)    
    # save the interpolated data
    os.chdir(originalDir_real)
    np.savetxt('U1_FE.txt', U1_FE)
    np.savetxt('V1_FE.txt', V1_FE)
    np.savetxt('U_FE.txt', U_FE)
    np.savetxt('V_FE.txt', V_FE)
    # load the ground truth data generated by FEM in full sqaure
    U1_FE = np.loadtxt('U1_FE.txt')
    V1_FE = np.loadtxt('V1_FE.txt')
    U_FE = np.loadtxt('U_FE.txt')
    V_FE = np.loadtxt('V_FE.txt')

    plot_relative_error(X1, Y1, np.abs(s_u_pred_tot -U1_FE), 'u_error_coupling_11_outer', rf'$|u_{{\mathrm{{FE}}}} - u_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{11}}}|$')
    plot_relative_error(X1, Y1, np.abs(s_v_pred_tot -V1_FE), 'v_error_coupling_11_outer', rf'$|v_{{\mathrm{{FE}}}} - v_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{11}}}|$')
    plot_relative_error(X, Y, np.abs(u - U_FE), 'u_error_coupling_11_inner', rf'$|u_{{\mathrm{{FE}}}} - u_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{11}}}|$')
    plot_relative_error(X, Y, np.abs(v - V_FE), 'v_error_coupling_11_inner', rf'$|v_{{\mathrm{{FE}}}} - v_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{11}}}|$')

except Exception as e:
    print(f"Failed to open the directory with ground truth data generated by FEM")










        




