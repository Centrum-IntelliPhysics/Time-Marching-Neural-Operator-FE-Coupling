# Table of Content 
-[General information](#general-information)

-[Application](#application)

-[Method](#Method)

-[Content](#Content)

-[Get started](#Get-started)

-[Contact](#contact)

# General information
This Git repository contains codes for the '**Time-marching neural operator–FE coupling: AI-accelerated physics modeling**' paper published in [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2025.118319)

Authors: [Wei Wang](https://scholar.google.com/citations?user=t1RXEkgAAAAJ&hl=zh-CN), [Maryam Hakimzadeh](https://scholar.google.com/citations?user=kff1AN0AAAAJ&hl=en), [Haihui Ruan](https://scholar.google.com/citations?user=TXDuvWMAAAAJ&hl=zh-CN), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra)
## initial motivations
The motivation is to establish a FE-NO coupling framework using a Schwartz alternating method to solve complex and nonlinear regions by a pretrained DeepONet, while the remainder is handled by a FEM. FEM requires extremely fine meshes to capture the nonlinear behaviors, which is computationally expensive; while DeepONets are non-linear mappings and can be pretrained offlines, which cost almost no time during the simulations. The combination of the two methods are expected to significantly improve the computational efficiency while maintaining the accuracy of the solution.
![schematic_DD](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schematic_domain_decomposition.png)
# Application 
## Elasto-dynamic 
Note that yellow square indicates the NO domain. The below GIF shows the plane wave propagation in FE and FE-NO coupling framework, along with the error evolution versus time step. 
![elasto_dynamic_results](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Elasto-dynamic/Elasto_dynamic_GIF.gif)

## linear elasticity and hyper elasticity
The results for linear staticity under static loading and hyper-elasticity under quasi-static loading are shown in folder [linear staticity static loading
](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Linear%20Elasticity%20Static%20loading) and [hyper-elasticity quasi-static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Hyper-elasticity%20quasi-static%20loading), respectively.

# Method
To achieve FE-NO coupling in dynamic problems, it requires both the spatial and temporal dimension coupling. The spatial coupling is achieved by a Schwartz alternating method at overlapping boundary, while the temporal coupling is achieved by Newmark-beta method integration in DeepONet. While in static or quasi-static problems, only the spatial coupling is needed. 

## Schwartz alternating method at overlapping boundary
![Schwartz_alternating_method](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schwartz_alternating_method.png)
## DeepONet structures 

![NO_structures](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/NO_structure.png)
## Time-marching workflow
![time-marching workflow](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/time-marching_workflow.png)

# Content 

# Get started  
Create conda environment and install [FEniCSx](https://fenicsproject.org/download/) (a FEM solver)   
Conda install:
<pre><code>conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
</code></pre>

Install the [JAX](https://docs.jax.dev/en/latest/installation.html)
CUDA version:
<pre><code>pip install --upgrade pip
# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
</code></pre>  



# Citation 
If you find this Github repository useful for your work, please condier citing this work:
<pre><code>
</code></pre> 
# Contact 
For more information or questions please contact 

-[Wei Wang@jh.edu](mailto:wwang198@jh.edu)

-[Somdatta Goswami@jhu.edu](mailto:somdatta@jhu.edu)






