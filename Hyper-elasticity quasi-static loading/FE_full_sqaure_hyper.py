from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
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
from ufl import dx
from Hyper_utils import plot_disp, createFolder
import dolfinx 


#region Save path       
originalDir = os.getcwd()
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'hyper_elasticity_quasi_static_ground_truth'  
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
num_points1 = 200  


#region FEM 
os.chdir('Hyper_elastic_Gmsh')
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("full_sqaure.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 1, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
os.chdir(origin_real)

### linear materials parameters 
E = 0.210e-2 #10GPa
nu = 0.3 

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))
# find the boundaries and set the boundary conditions
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1
# region Location BC
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)

def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)




def disk_out(x):
    xc = x[0][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 >= radius**2)]
    yc = x[1][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 >= radius**2)]
    return xc, yc 

def disk1(x):
    xc = x[0][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius**2)]
    yc = x[1][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius**2)]
    return xc, yc 

def disk2(x):
    xc = x[0][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius1**2)]
    yc = x[1][np.where((x[0]-center[0])**2 + (x[1]-center[1])**2 <= radius1**2)]
    return xc, yc

# region Mypression
#=====================================================================
# MyExpression is used to interpolate the displacement at the boundary
#=====================================================================
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


# strain function in ufl(a domain specific language for declaration of finite element discretizations of variational forms)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# stress function in ufl
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


    
# define the variational problem in FEM     
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0))) ## later T.value[2] refer to this value 
v = ufl.TestFunction(V)
uh = fem.Function(V)
# Spatial dimension
d = len(uh)
# Identity tensor
I = ufl.variable(ufl.Identity(d))
# Deformation gradient
F_grad = ufl.variable(I + ufl.grad(uh))
# Right Cauchy-Green tensor
C = ufl.variable(F_grad.T * F_grad)
# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F_grad))
# Elasticity parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress (first Piola–Kirchhoff stress tensor)
P = ufl.diff(psi, F_grad)
# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx

# out hole boundary
u_geometry = mesh1.geometry.x
X, Y = u_geometry[:,0], u_geometry[:,1]
x_out, y_out = disk_out(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_disk1, y_disk1 = disk1(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))
x_disk2, y_disk2 = disk2(np.vstack([X.reshape(1,-1), Y.reshape(1,-1)]))

bt_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bt_points)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

left_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], -1.))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_points)
bc_l = fem.dirichletbc(uD, boundary_dofs, V)

niter =1000
ts_tot = 5
#### Coupling  
for ts in range(0, ts_tot):

    top_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
    uD_top = np.array([0, 0.05*(ts+1), 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
    bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)
    
    right_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, right)
    uD_right = np.array([0.05*(ts+1), 0 , 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, right_b)
    bc_r = fem.dirichletbc(uD_right, boundary_dofs, V)

    error_list = []
    break_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    #### Coupling  
    bcs  = [bc_top, bc_l, bc_r, bc_bt] 
    problem = NonlinearProblem(F, uh, bcs)
    solver = NewtonSolver(mesh1.comm, problem)
    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"
    num_its, converged = solver.solve(uh)
    assert (converged)
    uh.x.scatter_forward()
    print(f"Time step {ts}, Number of iterations {num_its}, disp {uD_top}")

    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]


    plot_disp(X,Y,U_,'displacement u ts=' +str(ts), rf'$u_{{\mathrm{{FE}}}}^{{{ts}}}$')
    plot_disp(X,Y,V_,'displacement v ts=' +str(ts), rf'$v_{{\mathrm{{FE}}}}^{{{ts}}}$')

    np.savetxt('u ts=' +str(ts) +' .txt', U_)
    np.savetxt('v ts=' +str(ts) +' .txt', V_)
    np.savetxt('X.txt', X)
    np.savetxt('Y.txt', Y)

        










        




