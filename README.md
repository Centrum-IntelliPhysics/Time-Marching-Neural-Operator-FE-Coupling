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
## Initial motivations
The primary motivation is to establish a FE-NO coupling framework based on domain decomposition method to solve complex and nonlinear regions by a pretrained DeepONet, while the remainder is handled by a FEM. The FEM requires extremely fine meshes to capture the highly nonlinear behaviors, which is computationally expensive; while DeepONets are non-linear mappings and can be pretrained offlines, which cost negligible time during the simulations.

The coupling of the two solvers is expected to leverage their complementary strengths, leading to a significant enhancement in computational efficiency and robustness while maintaining solution accuracy.
![schematic_DD](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schematic_domain_decomposition.png)
# Application 
## Elasto-dynamic 
The yellow square denotes the NO domain. The GIF below illustrates plane wave propagation within the FE and FE-NO coupling frameworks, alongside the temporal evolution of the error. The error does not grow exponentially but evolves erratically, indicating the limiting influence of autoregressive error accumulation.
![elasto_dynamic_results](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Elasto-dynamic/Elasto_dynamic_GIF.gif)

## Linear elasticity and hyper elasticity
The results for linear staticity under static loading and hyper-elasticity under quasi-static loading are shown in folder [linear staticity static loading
](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Linear%20Elasticity%20Static%20loading) and [hyper-elasticity quasi-static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Hyper-elasticity%20quasi-static%20loading), respectively.

# Method
To achieve FE-NO coupling in dynamic problems, it requires both the spatial and temporal dimension coupling. The spatial coupling is achieved by a Schwartz alternating method at overlapping boundary, while the temporal coupling is achieved by Newmark-beta method integration in DeepONet (i.e., time-marching DeepONet). While in static or quasi-static problems, only the spatial coupling is needed. 

## Schwartz alternating method at overlapping boundary
![Schwartz_alternating_method](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schwartz_alternating_method.png)
## Time-marching DeepONet structures 
The time-marching DeepONet structure is inspired by the Newmark-beta method, which consists of two branch networks and one trunk network: **branch1** encodes the displacement boundary condition at current time step, **branch2** encodes the displacement and velocity across the domain at previous time step, and the **trunk** network only encodes the spatial coordinates. Such DeepONets can be trained directly using residual and boundary loss functions, without requiring additional data, resulting in a physics-informed DeepONet architecture.
![NO_structures](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/NO_structure.png)
## Time-marching workflow
The implementation of the time-marchiing DeepONet is illustrated below:
![time-marching workflow](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/time-marching_workflow.png)

# Content 
In this respository, we provide the codes for the following problems:
- [Elasto-dynamic](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Elasto-dynamic)
- [Linear staticity under static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Linear%20Elasticity%20Static%20loading)
- [Hyper-elasticity under quasi-static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Hyper-elasticity%20quasi-static%20loading)

Each problem folder contains 'FE_full' file to run the standalone FEM simulation, 'prepare_DeepONet' file to train the specific DeepONet, 'FE_DeepONet' file to run the FE-NO coupling simulation, and 'README.md' file to provide the excutation order of the codes and the description of the results.

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
For more information or questions please contact: 

-[Wei Wang@jh.edu](mailto:wwang198@jh.edu)

-[Somdatta Goswami@jhu.edu](mailto:somdatta@jhu.edu)

The FE-NO coupling framework is currently under development. We warmly welcome any suggestions and feedback, and we are open to collaborating with researchers from diverse fields!


