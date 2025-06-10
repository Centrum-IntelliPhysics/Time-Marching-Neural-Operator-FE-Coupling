import meshio
import gmsh 
import os 
from Hyper_utils import createFolder
import math 
import matplotlib.pyplot as plt
import numpy as np

#region Save path       
originalDir = os.getcwd()
print('current path:', originalDir)
os.chdir(os.path.join(originalDir))

foldername = 'Hyper_elastic_Gmsh'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/')) 


lc = 0.02 
center = (0.0, 0.5, 0.0)  # Center of the circle
radius = 0.3             # Radius of the circle
radius1 = 0.3 - 0.05
num_points = 200      # Number of points on the circumference #so important !!!0 
num_points1 = 200  


# region outer region 
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

#gmsh.model.mesh.setRecombine(2, inner_square)  # choose quadrilateral elements
#gmsh.model.mesh.setRecombine(2, background)  # choose quadrilateral elements
# insert the inner square into the background
# Fragment the surfaces
out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(
    [(2, background)],  # Target entities
    [(2, surface_hole1)]  # Tool entities
)

# Synchronize after boolean operation
gmsh.model.occ.synchronize()

# Set mesh size
#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.2)

# set quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[0][1])  # choose quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[1][1])  # choose quadrilateral elements

# Generate 2D mesh


# Get all surfaces after fragmentation
surfaces = gmsh.model.getEntities(2)

# Create physical groups for the two regions
# Now we use the actual tags from the fragment operation
outer_region = gmsh.model.addPhysicalGroup(2, [background], tag=1)
inner_region = gmsh.model.addPhysicalGroup(2, [surface_hole1], tag=2)

gmsh.model.mesh.generate(2)

# the boundary physical group should be fore lines physical group ??? 

#gmsh.model.setPhysicalName(2, outer_region, "Outer_Square")
#gmsh.model.setPhysicalName(2, inner_region, "Inner_Square")

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

# Set mesh size
#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.2)

# set quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[0][1])  # choose quadrilateral elements
#gmsh.model.mesh.setRecombine(2, out_dim_tags[1][1])  # choose quadrilateral elements

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

# full square 
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
gmsh.write("full_square.msh")

# Finalize GMSH
gmsh.finalize()

msh = meshio.read("full_square.msh")
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
plt.savefig('full_sqaure' + ".jpg", dpi=700)
plt.show()
