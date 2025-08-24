# Table of Content 
-[General information](#general-information)

-[Applications](#applications)

-[Methods](#Methods)

-[Content](#Content)

-[Get started](#Get-started)

-[Contact](#contact)

-[Data Availability](#Data-Availability)

# General information
This Git repository contains codes for the '**Time-marching neural operator–FE coupling: AI-accelerated physics modeling**' paper published in [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2025.118319)

Authors: [Wei Wang](https://scholar.google.com/citations?user=t1RXEkgAAAAJ&hl=zh-CN), [Maryam Hakimzadeh](https://scholar.google.com/citations?user=kff1AN0AAAAJ&hl=en), [Haihui Ruan](https://scholar.google.com/citations?user=TXDuvWMAAAAJ&hl=zh-CN), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra)
## Initial motivations
The primary objective is to develop an FE–NO coupling framework using domain decomposition framework. In this framework, a pretrained deep operator network (DeepONet) is employed to efficiently resolve complex, nonlinear subdomains specifically replacing the locations where fine meshes are requires, while the Finite Element Method (FEM) handles the remaining regions. Capturing strongly nonlinear behaviors with FEM alone is computationally expensive. In contrast, DeepONet represent nonlinear mappings that can be pretrained offline, making their evaluation during simulation essentially negligible in cost. In this work, the DeepONet is trained with the physics of the system only (no data is used).

The coupling of the two solvers is expected to leverage their complementary strengths, leading to a significant enhancement in computational efficiency and robustness while maintaining solution accuracy.
![schematic_DD](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schematic_domain_decomposition.png)
# Applications 
## Elasto-dynamic 
The yellow square marks the NO domain. The GIF below illustrates plane wave propagation in both the FE framework (serving as the ground truth) and the FE–NO coupling framework, along with the corresponding error evolution. The error remains bounded (within 2.5%) and does not grow monotonically; instead, it fluctuates, suggesting a limiting effect on autoregressive error accumulation.
![elasto_dynamic_results](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Elasto-dynamic/Elasto_dynamic_GIF.gif)

## Linear elasticity and hyper elasticity
The results for linear elasticity under static loading and hyperelasticity under quasi-static loading are provided in the folder [linear elasticity static loading
](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Linear%20Elasticity%20Static%20loading) and [hyper-elasticity quasi-static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Hyper-elasticity%20quasi-static%20loading), respectively.

# Methods
To achieve FE-NO coupling in dynamic problems, it requires both the spatial and temporal dimension coupling. The spatial coupling is achieved by a Schwarz alternating method at overlapping boundary, while the temporal coupling is achieved by Newmark-beta method integrated DeepONet (i.e., time-marching DeepONet). While in static or quasi-static problems, only the spatial coupling is needed. 

## Schwarz alternating method at overlapping boundary
![Schwarz_alternating_method](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/Schwarz_alternating_method.png)
## Time-marching DeepONet structures 
The time-marching DeepONet structure is inspired by the Newmark-beta method, which consists of two branch networks and one trunk network: **branch1** encodes the displacement boundary condition at current time step, **branch2** encodes the displacement and velocity across the domain at previous time step, and the **trunk** network only encodes the spatial coordinates. Such DeepONets can be trained directly using residual and boundary loss functions, resulting in Physics-Informed DeepONets. 
![NO_structures](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/NO_structure.png)
## Time-marching workflow
The implementation of the time-marching DeepONet coupled with numerical solver is illustrated below:
![time-marching workflow](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/blob/main/Readme_figures/time-marching_workflow.png)

# Content 
In this respository, we provide the codes for the following problems:
- [Elasto-dynamic](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Elasto-dynamic)
- [Linear elasticity under static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Linear%20Elasticity%20Static%20loading)
- [Hyper-elasticity under quasi-static loading](https://github.com/Centrum-IntelliPhysics/Time-Marching-Neural-Operator-FE-Coupling/tree/main/Hyper-elasticity%20quasi-static%20loading)

Each problem folder contains `FE_full` file to run the standalone FEM simulation, `prepare_DeepONet` file to train the specific DeepONet, `FE_DeepONet` file to run the FE-NO coupling simulation, and `README.md` file to provide the execution order of the codes and the simulation results.

# Get started  
Create conda environment and install [FEniCSx](https://fenicsproject.org/download/) (an FEM solver)   
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

Version:
<pre><code>Python                        3.9.22
jax                           0.4.30
jax-cuda12-pjrt               0.4.30
jax-cuda12-plugin             0.4.30
jaxlib                        0.4.30
jaxtyping                     0.2.36
fenics-basix                  0.9.0
fenics-dolfinx                0.9.0
fenics-ffcx                   0.9.0
fenics-ufl                    2024.2.0
</code></pre>



# Citation 
If you find this Github repository useful for your work, please consider citing this work:
<pre><code>@article{WANG2025118319,
title = {Time-marching neural operator–FE coupling: AI-accelerated physics modeling},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {446},
pages = {118319},
year = {2025},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2025.118319},
url = {https://www.sciencedirect.com/science/article/pii/S0045782525005912},
author = {Wei Wang and Maryam Hakimzadeh and Haihui Ruan and Somdatta Goswami},
keywords = {Time marching, Physics-informed neural operator, Hybrid solver, Domain decomposition},
abstract = {Numerical solvers for partial differential equations (PDEs) often struggle to balance computational efficiency with accuracy, especially in multiscale and time-dependent systems. Neural operators offer a promising avenue to accelerate simulations, but their practical deployment is hindered by several challenges: they typically require large volumes of training data generated from high-fidelity solvers, tend to accumulate errors over time in dynamical settings, and often exhibit poor generalization in multiphysics scenarios. This work introduces a novel hybrid framework that integrates physics-informed deep operator network (PI-DeepONet) with finite element method (FEM) through domain decomposition and leverages numerical analysis for time marching. The core innovation lies in efficient coupling FEM and DeepONet subdomains via a Schwarz alternating method, expecting to solve complex and nonlinear regions by a pre-trained DeepONet, while the remainder is handled by conventional FEM. To address the challenges of dynamic systems, we embed the Newmark-Beta time-stepping scheme directly into the DeepONet architecture, substantially reducing long-term error propagation. Furthermore, an adaptive subdomain evolution strategy enables the ML-resolved region to expand dynamically, capturing emerging fine-scale features without remeshing. The framework’s efficacy has been rigorously validated across a range of solid mechanics problems—spanning static, quasi-static, and dynamic regimes including linear elasticity, hyperelasticity, and elastodynamics—demonstrating accelerated convergence rates (up to 20 % improvement in convergence rates compared to conventional FE coupling approaches) while preserving solution fidelity with error margins consistently below 3 %. Our extensive case studies demonstrate that our proposed hybrid solver: (1) reduces computational costs by eliminating fine mesh requirements, (2) mitigates error accumulation in time-dependent simulations, and (3) enables automatic adaptation to evolving physical phenomena. This work establishes a new paradigm for coupling state-of-the-art physics-based and machine learning solvers in a unified framework—offering a robust, reliable, and scalable pathway for high-fidelity multiscale simulations.}
}
</code></pre> 
# Contact 
For more information or questions please contact: 

-[Wei Wang@jh.edu](mailto:wwang198@jh.edu)

-[Somdatta Goswami@jhu.edu](mailto:somdatta@jhu.edu)

The FE-NO coupling framework is currently under development. We warmly welcome any suggestions and feedback, and we are open to collaborating with researchers from diverse fields!

# Data Availability 
To train the DeepONet in elasto-dynamic, some data should be used, as provided in [Elasto_dynamic_dataset](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/wwang198_jh_edu/EpQ_n3R5_BJEi4dONwKmt6gBfWgHTd4Z-lShL8qXFKHgJQ?e=QSmSQ7).
