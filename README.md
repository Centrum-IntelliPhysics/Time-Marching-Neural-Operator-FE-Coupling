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
# Contact 





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


