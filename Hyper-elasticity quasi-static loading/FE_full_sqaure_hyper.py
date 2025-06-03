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
#### utils 

def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)



def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


# region Plot funcs
def plot_disp(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(20)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=20)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('x', fontsize=24)
    ax.set_ylabel('y', fontsize=24)
    ax.set_title(title, fontsize=36, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=18, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()


def plot_deformation_uy(u, V2, foldername):
    # Start virtual framebuffer if off-screen rendering is needed
    pyvista.start_xvfb()
    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(V2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    grid["uy"] = u.x.array.reshape((geometry.shape[0], 3))[:, 1]
    # Add the grid to the plotter
    #actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    
    warped = grid.warp_by_vector("u", factor=1)
    #actor_1 = p.add_mesh(warped, show_edges=True)
    actor_1 = p.add_mesh(warped, scalars= 'uy', cmap='seismic', show_edges= False) ## warped for deformation
    p.remove_scalar_bar()
    ## scalars for add values on the mesh
    p.add_scalar_bar(
    vertical=True,
    position_x=0.8,
    position_y=0.15,
    label_font_size=20,
    title_font_size=22,      # Font size for the title
    height=0.7,              # Height of the colorbar (adjust based on your needs)
    width=0.07             # Width of the colorbar (adjust based on your needs))  # Adjust the font size of the labels  # Move colorbar to the right side
    )
    p.show_axes()
    #p.add_mesh(grid, show_edges=True)

    # Adjust the view to align with the XY plane
    p.view_xy()

    # Show the plot (necessary for off-screen rendering)
    p.show()

    # Save the screenshot
    p.screenshot(foldername + ".png")

def plot_deformation_u(u, V2, foldername):
    # Start virtual framebuffer if off-screen rendering is needed
    pyvista.start_xvfb()
    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(V2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
    # Add the grid to the plotter
    #actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1)
    #actor_1 = p.add_mesh(warped, show_edges=True)
    actor_1 = p.add_mesh(warped, scalars= "u", cmap="seismic", show_edges= False) ## warped for deformation
    ## scalars for add values on the mesh

    p.show_axes()
    #p.add_mesh(grid, show_edges=True)

    # Adjust the view to align with the XY plane
    p.view_xy()

    # Show the plot (necessary for off-screen rendering)
    p.show()

    # Save the screenshot
    p.screenshot(foldername + ".png")

#region Save path       
originalDir = os.getcwd()
print('curent working directory:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'FE_full_square_hyper'  
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


# region Hole 
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
    [(2,background),(2, surface_hole1), (2, surface_hole)],  # Target entities
    []  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Set mesh size
#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.2)

# set quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[0][1])  # choose quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[1][1])  # choose quadrilateral elements

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
gmsh.model.addPhysicalGroup(1, lines1 , name="InnerSquareEdges")# facet tag = 3
gmsh.model.addPhysicalGroup(1, lines, name="HoleEdges") # facet tag = 4

# Write the mesh to a file (optional)
gmsh.write("3_point_bending_inner1.msh")

# Finalize GMSH
gmsh.finalize()


import meshio

# 读取Gmsh生成的网格文件 (.msh)
msh = meshio.read("3_point_bending_inner1.msh")

# 提取点和单元信息
points = msh.points
cells = msh.cells_dict["triangle"]  
# 使用三角形单元

# 创建图形
plt.figure(figsize=(8, 8))

# 绘制每个三角形
for cell in cells:
    polygon = points[cell]
    # 将多边形闭合
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5)  # 用黑色线条绘制

# 设置坐标轴
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization2')
plt.savefig('Gmsh Mesh Visualization2' + ".jpg", dpi=700)
plt.show()

#region FEM 
#### FEM  
import dolfinx 
# 导入 Gmsh 生成的 .msh 文件 
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("3_point_bending_inner1.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 1, (mesh1.geometry.dim, ))) ###without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 


### linear materials parameters 
E = 0.210e-2 #10GPa
nu = 0.3 

mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))
# 找到边界面
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1

# region Location BC
def bt(x):
    return np.isclose(x[1], -0.5)

def left(x):
    return np.isclose(x[0], -1.)
# Sub domain for rotation at top
def top(x):
    return np.isclose(x[1], 1.5) 

def right(x):
    return np.isclose(x[0], 1.)


circle = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))
#circle_1 = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True ))
#circle_rec = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0],-1.5) | np.isclose(x[0],2.5) | np.isclose(x[1], -0.5) | np.isclose(x[1], 1.5) )

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
class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_0  = Rbf(x0, y0, value[0])
        self.RBF_1  = Rbf(x0, y0, value[1])
        
    def eval(self, x):
        # Added some spatial variation here. Expression is sin(x)
        #print(x.shape)
        #print(x[0], x[1], x[2]) 
        
        values = np.zeros((self.V_dim, x.shape[1]))

        #values[0] = self.RBF_0(x[0], x[1])
        #values[1] = self.RBF_1(x[0], x[1])
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        
        #AAA1 = x[0][np.where(np.isclose((x[0]-center[0])**2 +
        #                          (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]
        
        #np.savetxt('AA', AAA1)
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
        # Added some spatial variation here. Expression is sin(x)
        #print(x.shape)
        #print(x[0], x[1], x[2]) 
        
        values = np.zeros((self.V_dim, x.shape[1]))

        #values[0] = self.RBF_0(x[0], x[1])
        #values[1] = self.RBF_1(x[0], x[1])
        values[0] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4), 
                            self.RBF_0(x[0], x[1]), 0)
        values[1] = np.where(np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius1**2, 1e-4, 1e-4),
                            self.RBF_1(x[0], x[1]), 0)
        return values 


    
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
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F_grad)

# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P) * dx
### ds will be redefined later


### define the inner boundary for handshake # the new defined ds 
boundaries = [(1, lambda x: np.isclose((x[0]-center[0])**2 + (x[1]-center[1])**2, radius**2, 1e-3, 1e-3))]

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


niter =1000
theta = 0.5 # relaxation coefficient
ts_tot = 5

#### Coupling  
for ts in range(0, ts_tot):

    top_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, top)
    uD_top = np.array([0, 0.1*(ts+1), 0], dtype=default_scalar_type)
    mesh1.topology.create_connectivity(fdim, tdim)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, top_b)
    bc_top = fem.dirichletbc(uD_top, boundary_dofs, V)
    error_list = []
    break_list = []
    u_list, v_list, u2_list, v2_list = [], [], [], []
    #### Coupling  

    bcs = [bc_bt, bc_top] 
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
    plot_deformation_uy(uh, V, 'deformation uy outer ts=' +str(ts))

    np.savetxt('u ts=' +str(ts) +' .txt', U_)
    np.savetxt('v ts=' +str(ts) +' .txt', V_)
    np.savetxt('X.txt', X)
    np.savetxt('Y.txt', Y)

        










        




