# Table of Content 
-[General information](#-General-information)

-[Method](#-Method)

-[Application](#-Application)

-[Content](#-Content)

-[Get started](#-Get-started)

-[Contact](#-contact)

# General information
This Git repository contains codes for the 'Time-marching neural operator–FE coupling: AI-accelerated physics modeling' paper published in [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2025.118319)

Authors: [Wei Wang](https://scholar.google.com/citations?user=t1RXEkgAAAAJ&hl=zh-CN), [Maryam Hakimzadeh](https://scholar.google.com/citations?user=kff1AN0AAAAJ&hl=en), [Haihui Ruan](https://scholar.google.com/citations?user=TXDuvWMAAAAJ&hl=zh-CN), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra)
# Method


# Application 
# Content 
# Get started 

# Citation 
If you find this Github repository useful for your work, please condier citing this work:
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
For more information or questions please contact 





# DeepONet_FEM_Coupling
Attached code in relevant paper: [Accelerating Multiscale Modeling with Hybrid Solvers: Coupling FEM and Neural Operators with Domain Decomposition](
https://doi.org/10.48550/arXiv.2504.11383)

## Virtual Environment   
Install the JAX on website: https://docs.jax.dev/en/latest/installation.html  
CUDA version:
<pre><code>pip install --upgrade pip
# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12]"
</code></pre>  


Install FEniCSx on website: https://fenicsproject.org/download/  
Conda install:
<pre><code>conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
</code></pre>


