'''
======================================================================
The FE-DeepONet or FE-NO coupling framework for hyper elastic materials 
under quasi-static loading conditions.
----------------------------------------------------------------------
The overlapping boundary is used. 
The domain decomposition and Schwartz alternating method is used by
exchanging the displacement at the overlapping boundary.

Two tractions (displacements) are applied on the top and right edege
======================================================================
'''
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import os 
import jax
import jax.numpy as jnp
from jax import grad, vmap
import dolfinx
from dolfinx import fem, default_scalar_type, la
from dolfinx.fem import functionspace 
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import Constant, Function 
from mpi4py import MPI
import numpy as np
from dolfinx.mesh import create_box, create_unit_square
import ufl 
from ufl import dx
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, create_matrix
from petsc4py import PETSc
import os 
from tqdm import trange 
import gmsh 
import math
from dolfinx import plot
from scipy.interpolate import Rbf, interp1d, griddata
import logging 
import meshio
import pyvista
import pickle
import time
import dolfinx 
from Hyper_utils import createFolder, plot_disp, plot_relative_error



#region Save path       
originalDir = os.getcwd()
print('current path:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_DeepONet_hyper_elasticity_quasi_static_coupling_results'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 

originalDir_real = os.path.join(originalDir, './'+ foldername + '/') 

# region gmsh
# Define circle parameters
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
radius1 = 0.3 - 0.05
num_points = 200      # Number of points on the circumference #so important !!!0 
num_points1 = 200  


#========================================================
# Use FEM to solve the outer region for hyper-elasticity
#========================================================

#region FEM 
# load the gmsh file
os.chdir('Hyper_elastic_Gmsh')
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("outer_region.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 1, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 
os.chdir(originalDir_real)
### materials parameters 
E = 0.210e-2
nu = 0.3 

# strain function in ufl(a domain specific language for declaration of finite element discretizations of variational forms)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

# stress function in ufl
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

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

# the fixed boundaries for bottom edge and left edge 
bottom_point = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], -0.5))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, bottom_point)
bc_bt = fem.dirichletbc(uD, boundary_dofs, V)

left_points = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], -1.))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_points)
bc_l = fem.dirichletbc(uD, boundary_dofs, V)


# the functions to find the outer cricle and inner circle for overlapping boundaries
# input: [2, n] --> [2, m] m points on the circular boundaries 
def on_cricle(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4))]
    return xc, yc 

def on_cricle_inner(x):
    xc = x[0][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))]
    yc = x[1][np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))]
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
# for the inner region receiving the boundary displacement from FEM
#=========================================================
from Prepare_DeepONet_hyper_ealstic_quasi_static import PI_DeepONet

m = num_points1 
d = 2 # dimension of the input data
ela_model = dict()
ela_model['E'] = E #1000 
ela_model['nu'] = nu

branch_layers =  [2*m, 100, 100, 100, 100, 800]
trunk_layers =  [d, 100, 100, 100, 100, 800]
model = PI_DeepONet(branch_layers, trunk_layers, **ela_model)
# 1211 is for disk case and it is correct
os.chdir(os.path.join(originalDir, './' + 'Pretrained_hyper_elasticity_quasi_static' + '/'))
print(os.getcwd())
with open('DeepONet_hyper_elasticity_quasi_static.pkl', 'rb') as f:
    params = pickle.load(f)

X11 = np.linspace(center[0]-radius-0.05, center[0]+radius+0.05, m)
Y11 = np.linspace(center[1]-radius-0.05, center[1]+radius+0.05, m)
X1_, Y1_ = np.meshgrid(X11, Y11)

os.chdir(originalDir_real)

# resort the sensor points
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


time_0 = time.time()
#region Coupling 
niter =1000
ts_tot = 5 # total quasi-static step
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
    for iter in range(0, niter):
        ## FEM 
        if iter == 0 and ts == 0:
            # first hyper elastic solver     
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
            print(f"Time step {ts}, inner_iter{iter}, Number of iterations {num_its}, disp {uD_top}")
        
        else:     
            # second hyper elastic solver  
            problem = NonlinearProblem(F, uh, bcs)
            solver = NewtonSolver(mesh1.comm, problem)
            # Set Newton solver options
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.convergence_criterion = "incremental"
            num_its, converged = solver.solve(uh)
            assert (converged)
            uh.x.scatter_forward()
            print(f"Time step {ts}, inner_iter{iter}, Number of iterations {num_its}, disp {uD_top}")
            print('success initiation')
        
        
        if ts >= ts_tot - 1 and iter == 0:
            start1 = time.time()
            
       
        u_values = uh.x.array.real
        u_tot = u_values.reshape(-1,3)
        U_, V_, W_= u_tot[:,0], u_tot[:,1], u_tot[:,2]
        # the displacement obtained from outer region (FEM) at the outer circle of overlapping boundary 
        u_c = U_[coor_r_out].reshape(1,-1)
        v_c = V_[coor_r_out].reshape(1,-1)
        
        gdim = mesh1.geometry.dim
        ### stress is 3*3  (gdim, gdim)
        Function_space_for_sigma = fem.functionspace(mesh1, ("CG", 1, (gdim, gdim)))        
        expr= fem.Expression(sigma(uh), Function_space_for_sigma.element.interpolation_points())        
        sigma_values = Function(Function_space_for_sigma) 
        sigma_values.interpolate(expr)
        sigma_tot = sigma_values.x.array.reshape(-1,9)
        
        
        if iter >= 0:
            # region handshake

            # Get the boundary dispalcement by using Rbf interpolation 
            uc_fun = Rbf(x_c_out, y_c_out, u_c)
            vc_fun = Rbf(x_c_out, y_c_out, v_c)

            u2c = uc_fun(x_c1_out, y_c1_out)[index].reshape(1,-1)
            v2c = vc_fun(x_c1_out, y_c1_out)[index].reshape(1,-1)

            # predict the displacement at the inner boundary
            #=================================================
            # the u_test and v_test are the displacement at
            # the outer boundary as boundary condition
            # the hc_test is the coordinate of the inner boundary
            #=================================================
            u_test = np.hstack([u2c, v2c]).reshape(1,-1) 
            v_test = np.hstack([u2c, v2c]).reshape(1,-1)
            hc_test = np.hstack([X1[coor_r1_in].reshape(-1,1), Y1[coor_r1_in].reshape(-1,1)])
            s_uc_pred_in, s_vc_pred_in = model.predict_s(params, u_test, v_test, hc_test)

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
            #print(fdim2)
            c_line_in = dolfinx.mesh.locate_entities_boundary(mesh1, fdim2, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4))
            uD_c2 = Function(V)
            uD_c2_value = np.vstack([u_c2,v_c2])
            # use MyExpression_inner_hole to interpolate the displacement at the inner boundary (from inner hole to outer region)
            uD_c2_fun = MyExpression_inner_hole(X1[coor_r1_in], Y1[coor_r1_in], uD_c2_value, mesh1.geometry.dim)
            uD_c2.interpolate(uD_c2_fun.eval)        
            boundary_dofs2 = fem.locate_dofs_topological(V, fdim2, c_line_in)
            bc_c2 = fem.dirichletbc(uD_c2, boundary_dofs2)            
            bcs  = [bc_top, bc_l, bc_r, bc_bt, bc_c2]    
            h_test = np.hstack([X1.reshape(-1,1), Y1.reshape(-1,1)])
            s_u_pred_tot, s_v_pred_tot = model.predict_s(params, u_test, v_test, h_test)
            U2_, V2_ = s_u_pred_tot.reshape(-1,1), s_v_pred_tot.reshape(-1,1)


        u_list.append(U_)
        v_list.append(V_)
        u2_list.append(U2_)
        v2_list.append(V2_)
        break_list.append(sigma_tot[:,0])
        if len(u_list) > 1:
            uv_L2 = np.linalg.norm(np.sqrt((u_list[-1] - u_list[-2])**2 + (v_list[-1] - v_list[-2])**2)) 
            uv_L2_2 = np.linalg.norm(np.sqrt((u2_list[-1] - u2_list[-2])**2 + (v2_list[-1] - v2_list[-2])**2))
            print('\n' ,'error', uv_L2 + uv_L2_2)
            error_list.append(uv_L2 + uv_L2_2)
            if  error_list[-1] < 1e-3:
        
                plot_disp(X,Y,U_,'displacement u ts=' +str(ts) +' iter=' + str(iter),rf'$u_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X,Y,V_,'displacement v ts=' +str(ts) +' iter=' + str(iter),rf'$v_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                np.savetxt('u ts=' +str(ts) +' .txt', U_)
                np.savetxt('v ts=' +str(ts) +' .txt', V_)
                np.savetxt('u2 ts=' +str(ts) +' .txt', U2_)
                np.savetxt('v2 ts=' +str(ts) +' .txt', V2_)
                plot_disp(X1,Y1,U2_,'displacement u inner ts=' +str(ts) +' iter=' + str(iter), rf'$u_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                plot_disp(X1,Y1,V2_,'displacement v inner ts=' +str(ts) +' iter=' + str(iter), rf'$v_{{\mathrm{{FE-NO}}}}^{{{ts}}}$')
                U2_tot = np.hstack([U2_.reshape(-1,1), V2_.reshape(-1,1),np.zeros((V2_.shape[0],1))])
                np.savetxt(f'error_list_FE_NN_hyper ts=' + str(ts) + ' iter=' + str(iter) + '.txt', error_list)
                # L2 error is smaller than 1e-3, break the loop
                break

np.savetxt('X.txt', X)
np.savetxt('Y.txt', Y)
np.savetxt('X1.txt', X1)
np.savetxt('Y1.txt', Y1)

time_1 = time.time()
print('time tot', time_1 - time_0)
print('time 4', time_1 - start1)
np.savetxt('consuming time',np.array([time_1 - time_0, time_1 - start1]))

# region plot error
#=========================================================
# Plot the error between the predicted displacement and the ground truth
# Ground truth data is generated by FEM in full square
#=========================================================

try:
    os.chdir(os.path.join(originalDir, './' + 'FE_DeepONet_hyper_elasticity_quasi_static_coupling_results' + '/'))
    X_NN, Y_NN, X1_NN, Y1_NN = np.loadtxt('X.txt'), np.loadtxt('Y.txt'), np.loadtxt('X1.txt'), np.loadtxt('Y1.txt')
    # for interpolation FE_NN coupling
    u_FE_NN_4, v_FE_NN_4 = np.loadtxt('u ts=' + str(i) + ' .txt'), np.loadtxt('v ts=' + str(i) + ' .txt')
    u2_FE_NN_4, v2_FE_NN_4 = np.loadtxt('u2 ts=' + str(i) + ' .txt'), np.loadtxt('v2 ts=' + str(i) + ' .txt')

    os.chdir(os.path.join(originalDir, './' + 'hyper_elasticity_quasi_static_ground_truth' + '/'))
    X_full, Y_full= np.loadtxt('X.txt'), np.loadtxt('Y.txt')
    # for interpolation FE_NN coupling
    u_full_4, v_full_4 = np.loadtxt('u ts=' + str(i) + ' .txt'), np.loadtxt('v ts=' + str(i) + ' .txt')
    start_0 = time.time()
    u_full_4_outer = Rbf(X_full, Y_full, u_full_4)(X_NN, Y_NN)
    v_full_4_outer = Rbf(X_full, Y_full, v_full_4)(X_NN, Y_NN)
    u_full_4_inner = Rbf(X_full, Y_full, u_full_4)(X1_NN, Y1_NN)
    v_full_4_inner = Rbf(X_full, Y_full, v_full_4)(X1_NN, Y1_NN)
    end_0 = time.time()
    print('time for interpolation:', end_0 - start_0)

    os.chdir(originalDir_real)
    np.savetxt('u_full_4_outer ts=' + str(i) + '.txt', u_full_4_outer)
    np.savetxt('v_full_4_outer ts=' + str(i) + '.txt', v_full_4_outer)
    np.savetxt('u_full_4_inner ts=' + str(i) + '.txt', u_full_4_inner)
    np.savetxt('v_full_4_inner ts=' + str(i) + '.txt', v_full_4_inner)
    os.chdir(originalDir_real)
    u_full_4_outer = np.loadtxt('u_full_4_outer ts=' + str(i) + '.txt')
    v_full_4_outer = np.loadtxt('v_full_4_outer ts=' + str(i) + '.txt')
    u_full_4_inner = np.loadtxt('u_full_4_inner ts=' + str(i) + '.txt')
    v_full_4_inner = np.loadtxt('v_full_4_inner ts=' + str(i) + '.txt')

    # plot the relative error between the FE_NN coupling and FE_FE coupling
    u_error_coupling_4 = np.abs(u_FE_NN_4 - u_full_4_outer)
    v_error_coupling_4 = np.abs(v_FE_NN_4 - v_full_4_outer)
    u2_error_coupling_4 = np.abs(u2_FE_NN_4 - u_full_4_inner)
    v2_error_coupling_4 = np.abs(v2_FE_NN_4 - v_full_4_inner)

    plot_relative_error(X_NN, Y_NN, u_error_coupling_4, 'u_FE_NO_error ts=' + str(i), rf'$|u_{{\mathrm{{FE}}}}^{{{i}}} - u_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{i}}}|$')
    plot_relative_error(X_NN, Y_NN, v_error_coupling_4, 'v_FE_NO_error ts=' + str(i), rf'$|v_{{\mathrm{{FE}}}}^{{{i}}} - v_{{\mathrm{{FE-NO}},\Omega_{{I}}}}^{{{i}}}|$')
    plot_relative_error(X1_NN, Y1_NN, u2_error_coupling_4, 'u2_FE_NO_error ts=' + str(i), rf'$|u_{{\mathrm{{FE}}}}^{{{i}}} - u_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{i}}}|$')
    plot_relative_error(X1_NN, Y1_NN, v2_error_coupling_4, 'v2_FE_NO_error ts=' + str(i), rf'$|v_{{\mathrm{{FE}}}}^{{{i}}} - v_{{\mathrm{{FE-NO}},\Omega_{{II}}}}^{{{i}}}|$')

except Exception as e:
    print(f"Failed to open the directory with ground truth data generated by FEM")






        




