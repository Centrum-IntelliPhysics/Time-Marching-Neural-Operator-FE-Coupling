import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# region utils 
#utils 
def plot_disp(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic')
    #scatter = ax.scatter(X, Y, c=u, cmap='viridis')  # Use 'viridis' colormap for better visibility

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('x', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()

def plot_relative_error(X, Y, u, foldername, title):
    '''
    Plot the relative error field
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    
    
    
    # Adjust scientific notation offset text (e.g., "1e-3")
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size
    cbar.ax.get_yaxis().offsetText.set_position((1.1, 0.5))  # Move offset text further right
    
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    cbar.ax.yaxis.offsetText.set_position((3.0, 0.5))  # Move offset text further right

    # Set font properties
    ax.set_xlabel('X', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold')
    
    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()

# The vmax and vmin are used to set the colorbar range
def plot_disp_real(X, Y, u, foldername, title):
    '''
    Plot the displacement field
    for bounded value 
    '''
    plt.figure(figsize=(10, 8.5))
    
    # Set a compact layout between the main plot and the colorbar
    ax = plt.gca()  # Get the current axis
    scatter = ax.scatter(X, Y, c=u, cmap='seismic',vmin=0, vmax=0.01)

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02, location='right')  # Reduce the padding
    cbar.formatter = ScalarFormatter()  # Set the default formatter
    cbar.formatter.set_scientific(True)  # Enable scientific notation
    cbar.formatter.set_powerlimits((-2, 2))  # Show scientific notation for numbers smaller than 0.01
    cbar.ax.get_yaxis().offsetText.set_fontsize(32)  # Set scientific notation size  # fontsize of formatter 12
    #cbar.ax.get_yaxis().offsetText.set_x(1.1)  # Move scientific notation to the right 
    cbar.ax.tick_params(labelsize=32)  # Adjust the font size of the colorbar labels
    # Set font properties
    ax.set_xlabel('X', fontsize=32, fontweight='bold')
    ax.set_ylabel('Y', fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=40, fontweight='bold', y = 1.05)  # Increase the distance between the title and the plot

    # Adjust tick size and font size
    ax.tick_params(axis='both', which='major', labelsize=32, length=10, width=2)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=32, length=5, width=1)  # Minor ticks

    # Adjust the subplot layout
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # The rect parameter controls the overall layout (reduce right margin)

    # Save and display
    plt.savefig(foldername + ".jpg", dpi=700, bbox_inches='tight')  # bbox_inches='tight' reduces the extra margins
    plt.show()
    plt.close()




def plot_error_list(error_list, foldername):
    """
    Plots the error list as a line figure with a logarithmic scale on the y-axis.

    Parameters:
    error_list (list or array-like): List of error values to plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(error_list, marker='o', linestyle='-', color='b', label='Error')
    plt.yscale('log')
    plt.title(foldername)
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()


def plot_mesh(cells, points, foldername):
    '''
    plot the mesh from gmsh
    '''

    plt.figure(figsize=(8, 8))

    for cell in cells:
        polygon = points[cell]
        #create a closed polygon
        polygon = np.vstack([polygon, polygon[0]])
        plt.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.2)

    plt.gca().set_aspect('equal')
    plt.xlabel('X', fontsize=32, fontweight='bold')
    plt.ylabel('Y', fontsize=32, fontweight='bold')
    plt.title(foldername, fontsize=40, fontweight='bold')
    plt.savefig( foldername + ".jpg", dpi=700)
    plt.show()

def plot_boundary(mesh, facet_values, facet_indices,colors, foldername, fdim = 1):
    '''
    plot the boundary of the mesh
    params: 
            mesh: the mesh object
            facet_values: the values of the facets
            facet_indices: the indices of the facets
            colors: the color map for the tags
            foldername: the foldername to save the plot
    
    '''

    plt.figure(figsize=(8, 8))
    # Loop through the unique tag values and plot the facets with the same color
    for tag in np.unique(facet_values):
        facet_indices_with_tag = facet_indices[facet_values == tag]
        color = colors.get(tag, 'black')  # Use 'black' if the tag is not in the color map
        for facet_idx in facet_indices_with_tag:
            # retrieve the dofs of the facet  (cause Lagrangian 1 order element CG 1)
            # DOF is the vertex of the mesh
            facet_dofs = mesh.topology.connectivity(fdim, 0).links(facet_idx)
            # retrieve the coordinates of the facet
            facet_coords = mesh.geometry.x[facet_dofs]
            plt.plot(facet_coords[:, 0], facet_coords[:, 1], color=color, label=f"Tag {tag}")
            plt.savefig(foldername)
    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
